"""Presentation-facing services for the optional Manager UI."""

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from ..adapters.registry import get_adapter
from ..core.adapter import (
    CapabilityUnavailable,
    ExtensionInstaller,
    ExtensionProvider,
    ExtensionVerifier,
    HardwareProvider,
    PythonPackageAdapter,
)
from ..core.backup import load_backup, load_extension_backup
from ..core.catalog import CatalogError, ensure_catalog, iter_candidates, load_catalog
from ..core.detection import TargetLayout
from ..core.hardware import detected_gfx_targets, hardware_from_devices, probe_hardware
from ..core.extension_resolver import run_extension_resolver
from ..core.install import (
    InstallResult,
    apply_extension_restore,
    apply_install,
    apply_restore,
    build_extension_restore_plan,
    build_restore_plan,
    dry_run_install,
)
from ..core.planning import validate_plan_binding


__all__ = ("ManagerService", "detected_gfx_targets")


@dataclass(frozen=True)
class CatalogState:
    path: Path
    source: str
    refreshed: bool
    catalog_sha256: str
    fetched_at: str | None = None
    artifact_count: int | None = None
    cache_age_seconds: float | None = None
    source_failure_count: int | None = None
    source_failures: tuple[str, ...] = ()
    timestamp_source: str = "unknown"


class ManagerService:
    """Keep UI orchestration thin and delegate policy to core and adapters."""

    def __init__(self, adapter_name="comfyui"):
        self.adapter_name = adapter_name
        self.adapter = get_adapter(adapter_name)
        self.target: TargetLayout | None = None
        self.catalog = None
        self.catalog_state = None

    def set_adapter(self, adapter_name):
        self.adapter_name = adapter_name
        self.adapter = get_adapter(adapter_name)
        self.target = None

    def detect(self, path):
        return self.adapter.detect(path)

    def commit_target(self, target):
        self.target = target
        return target

    def clear_target(self):
        self.target = None

    def load_catalog(self, path=None, *, base_url=None, refresh=False):
        catalog_path = ensure_catalog(path, base_url=base_url, refresh=refresh)
        catalog = load_catalog(catalog_path)
        catalog_bytes = catalog_path.read_bytes()
        catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
        try:
            catalog_index = json.loads(catalog_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CatalogError(f"invalid JSON catalog: {catalog_path}") from error
        source = base_url or ("local file" if path is not None else "official Matrix source")
        manifest = {}
        manifest_paths = []
        is_catalog_index = isinstance(catalog_index.get("artifacts"), list)
        if is_catalog_index:
            manifest_paths.append(catalog_path.parent / "source-manifest.json")
            if catalog_path.parent.name == "data":
                manifest_paths.insert(
                    0, catalog_path.parent.parent / "source-manifest.json"
                )
        for manifest_path in manifest_paths:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            break
        manifest_hash = manifest.get("catalog_sha256")
        if manifest_hash is not None and manifest_hash != catalog_sha256:
            raise CatalogError(
                f"catalog hash does not match source manifest: {catalog_path}"
            )
        fetched_at = manifest.get("fetched_at") or catalog.get("_catalog_generated_at")
        timestamp_source = "fetched_at" if manifest.get("fetched_at") else (
            "generated_at" if fetched_at else "unknown"
        )
        age = None
        if isinstance(fetched_at, str):
            try:
                observed = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
                age = max(0.0, (datetime.now(timezone.utc) - observed).total_seconds())
            except ValueError:
                age = None
        statuses = catalog.get("_collection_statuses")
        source_failures = []
        if isinstance(statuses, dict):
            for status in statuses.values():
                for result in status.get("results", ()) if isinstance(status, dict) else ():
                    if isinstance(result, dict) and result.get("status") == "failed":
                        source_failures.append(str(result.get("source_id") or "unknown"))
            failure_count = len(source_failures)
        else:
            failure_count = None
        state = CatalogState(
            path=catalog_path,
            source=source,
            refreshed=refresh,
            catalog_sha256=catalog_sha256,
            fetched_at=fetched_at,
            artifact_count=(
                manifest.get("artifact_count")
                if manifest.get("artifact_count") is not None
                else len(catalog_index.get("artifacts", ()))
                if isinstance(catalog_index.get("artifacts"), list)
                else None
            ),
            cache_age_seconds=age,
            source_failure_count=failure_count,
            source_failures=tuple(sorted(source_failures)),
            timestamp_source=timestamp_source,
        )
        return catalog, state

    def commit_catalog(self, result):
        self.catalog, self.catalog_state = result
        return result

    def clear_catalog(self):
        self.catalog = None
        self.catalog_state = None

    def candidates(
        self,
        *,
        platform,
        gfx,
        channel=None,
        rocm_version=None,
        include_unavailable=False,
        include_incompatible=False,
        distribution_family=None,
        lifecycle=None,
        candidate_kind=None,
    ):
        if self.catalog is None:
            raise ValueError("load a Matrix catalog before listing candidates")
        python_tag = None
        if isinstance(self.adapter, PythonPackageAdapter) and self.target is not None:
            python_tag = self.adapter.python_tag(self.target)
        return iter_candidates(
            self.catalog,
            platform=platform,
            gfx=gfx,
            channel=channel,
            rocm_version=rocm_version,
            python_tag=python_tag,
            include_unavailable=include_unavailable,
            include_incompatible=include_incompatible,
            distribution_family=distribution_family,
            lifecycle=lifecycle,
            candidate_kind=candidate_kind,
        )

    def inventory(self, candidate=None):
        self._require_target()
        return self.adapter.inventory(self.target, candidate)

    def prepare_target_snapshot(self, path):
        """Prepare read-only target package and extension evidence without committing state."""

        target = self.detect(path)
        return target, *self.prepare_detected_target_snapshot(target)

    def prepare_detected_target_snapshot(self, target):
        """Prepare read-only package and extension evidence for an already detected target."""

        inventory = self.adapter.inventory(target)
        extension_report = None
        if isinstance(self.adapter, ExtensionProvider):
            extension_report = self._extension_inventory_for_target(
                target,
                inventory=inventory,
            )
        return inventory, extension_report

    def verify(self):
        self._require_target()
        return self.adapter.verify(self.target)

    def hardware(self, runtime=None):
        target = self._require_target()
        runtime_devices = getattr(runtime, "devices", ()) if runtime is not None else ()
        if runtime_devices:
            return hardware_from_devices(
                runtime_devices,
                scope="target-runtime",
                source="application-runtime",
                target_root=target.root,
                host_platform=getattr(runtime, "host_platform", None),
            )
        if isinstance(self.adapter, HardwareProvider):
            return self.adapter.hardware(target)
        return probe_hardware(target)

    def host_hardware(self):
        """Probe host GFX before an application target has been selected."""

        return probe_hardware()

    def inspect(self):
        """Collect application runtime evidence and shared hardware evidence."""

        self._require_target()
        runtime = None
        runtime_error = None
        try:
            runtime = self.verify()
        except CapabilityUnavailable as error:
            runtime_error = str(error)
        hardware = self.hardware(runtime)
        return runtime, hardware, runtime_error

    def plan(self, candidate):
        self._require_target()
        return dry_run_install(
            self.target,
            candidate,
            catalog_hash=self.catalog_state.catalog_sha256 if self.catalog_state else None,
            adapter_id=self.adapter.id,
            target_gfx=candidate.get("gfx"),
        )

    def apply_core(self, result, *, allow_unverified=False, backup_dir=None):
        self._require_target()
        plan = result.plan if isinstance(result, InstallResult) else result
        candidate = plan.candidate
        validate_plan_binding(
            plan,
            self.target,
            candidate,
            catalog_hash=self.catalog_state.catalog_sha256 if self.catalog_state else None,
            adapter_id=self.adapter.id,
            target_gfx=candidate.get("gfx"),
        )
        return apply_install(
            self.target,
            candidate,
            backup=None,
            allow_unverified=allow_unverified,
            plan=plan,
            backup_dir=backup_dir,
            catalog_hash=self.catalog_state.catalog_sha256 if self.catalog_state else None,
            adapter_id=self.adapter.id,
            target_gfx=candidate.get("gfx"),
        )

    def restore_core(
        self, backup_path, *, apply=False, allow_network_restore=False
    ):
        self._require_target()
        backup = load_backup(
            backup_path,
            materialize=apply,
            allow_network_restore=allow_network_restore,
        )
        if apply:
            return apply_restore(self.target, backup)
        return InstallResult(plan=build_restore_plan(self.target, backup))

    def extension_plan(self, candidate, selections=()):
        self._require_target()
        if not isinstance(self.adapter, ExtensionProvider):
            raise ValueError(f"adapter {self.adapter.id} has no extension planning capability")
        profiles = (self.catalog or {}).get("_comfyui_extension_profiles", {})
        extension_catalog = (self.catalog or {}).get("_extension_catalog", {})
        return self.adapter.extension_plan(
            self.target,
            candidate,
            tuple(selections),
            profiles,
            extension_catalog,
        )

    def extension_capable(self):
        return isinstance(self.adapter, ExtensionProvider)

    def extension_inventory(self, candidate=None):
        self._require_target()
        return self._extension_inventory_for_target(self.target, candidate=candidate)

    def _extension_inventory_for_target(self, target, candidate=None, inventory=None):
        if not isinstance(self.adapter, ExtensionProvider):
            raise CapabilityUnavailable(f"adapter {self.adapter.id} has no extension inventory capability")
        profiles = (self.catalog or {}).get("_comfyui_extension_profiles", {})
        extension_catalog = (self.catalog or {}).get("_extension_catalog", {})
        return self.adapter.extension_inventory(
            target,
            candidate,
            profiles,
            extension_catalog,
            inventory=inventory,
        )

    def extension_resolve(self, candidate, selections=()):
        self._require_target()
        plan = self.extension_plan(candidate, selections)
        return run_extension_resolver(
            self.target,
            candidate,
            plan.extensions,
            selection_hash=plan.selection_hash,
        )

    def extension_verify(self, candidate, selections=()):
        self._require_target()
        if not isinstance(self.adapter, ExtensionVerifier):
            raise CapabilityUnavailable(f"adapter {self.adapter.id} has no extension verification capability")
        profiles = (self.catalog or {}).get("_comfyui_extension_profiles", {})
        extension_catalog = (self.catalog or {}).get("_extension_catalog", {})
        return self.adapter.extension_verify(
            self.target,
            candidate,
            tuple(selections),
            profiles,
            extension_catalog,
        )

    def apply_extensions(self, plan, backup_dir=None):
        self._require_target()
        if not isinstance(self.adapter, ExtensionInstaller):
            raise ValueError(f"adapter {self.adapter.id} has no extension installation capability")
        resolver_results = self.extension_resolve(plan.candidate, plan.selections)
        plan = replace(plan, resolver_results=resolver_results)
        backup = self.adapter.create_extension_backup(self.target, plan, backup_dir)
        return self.adapter.apply_extension_plan(self.target, plan, backup)

    def restore_extensions(self, backup_path, *, apply=False):
        self._require_target()
        backup = load_extension_backup(backup_path, materialize=apply)
        if apply:
            return apply_extension_restore(self.target, backup)
        return InstallResult(plan=build_extension_restore_plan(self.target, backup))

    def _require_target(self):
        if self.target is None:
            raise ValueError("detect a target before running this operation")
        return self.target
