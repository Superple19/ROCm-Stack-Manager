"""Presentation-facing services for the optional Manager UI."""

from dataclasses import dataclass
import json
from pathlib import Path

from ..adapters.registry import get_adapter
from ..core.adapter import (
    CapabilityUnavailable,
    ExtensionInstaller,
    ExtensionProvider,
    ExtensionVerifier,
    HardwareProvider,
    PythonPackageAdapter,
)
from ..core.backup import create_backup, load_backup, load_extension_backup
from ..core.catalog import ensure_catalog, iter_candidates, load_catalog
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


@dataclass(frozen=True)
class CatalogState:
    path: Path
    source: str
    refreshed: bool
    fetched_at: str | None = None
    catalog_sha256: str | None = None
    artifact_count: int | None = None


class ManagerService:
    """Keep UI orchestration thin and delegate policy to core and adapters."""

    def __init__(self, adapter_name="comfyui"):
        self.adapter_name = adapter_name
        self.adapter = get_adapter(adapter_name)
        self.target = None
        self.catalog = None
        self.catalog_state = None

    def set_adapter(self, adapter_name):
        self.adapter_name = adapter_name
        self.adapter = get_adapter(adapter_name)
        self.target = None

    def detect(self, path):
        self.target = self.adapter.detect(path)
        return self.target

    def load_catalog(self, path=None, *, base_url=None, refresh=False):
        catalog_path = ensure_catalog(path, base_url=base_url, refresh=refresh)
        self.catalog = load_catalog(catalog_path)
        source = base_url or ("local file" if path is not None else "official Matrix source")
        manifest = {}
        manifest_paths = [catalog_path.parent / "source-manifest.json"]
        if catalog_path.parent.name == "data":
            manifest_paths.insert(0, catalog_path.parent.parent / "source-manifest.json")
        for manifest_path in manifest_paths:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            break
        self.catalog_state = CatalogState(
            path=catalog_path,
            source=source,
            refreshed=refresh,
            fetched_at=manifest.get("fetched_at"),
            catalog_sha256=manifest.get("catalog_sha256"),
            artifact_count=manifest.get("artifact_count"),
        )
        return self.catalog, self.catalog_state

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

    def verify(self):
        self._require_target()
        return self.adapter.verify(self.target)

    def hardware(self, runtime=None):
        self._require_target()
        runtime_devices = getattr(runtime, "devices", ()) if runtime is not None else ()
        if runtime_devices:
            return hardware_from_devices(
                runtime_devices,
                scope="target-runtime",
                source="application-runtime",
                target_root=self.target.root,
                host_platform=getattr(runtime, "host_platform", None),
            )
        if isinstance(self.adapter, HardwareProvider):
            return self.adapter.hardware(self.target)
        return probe_hardware(self.target)

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
        backup = create_backup(self.target, backup_dir)
        return apply_install(
            self.target,
            candidate,
            backup,
            allow_unverified=allow_unverified,
            plan=plan,
            catalog_hash=self.catalog_state.catalog_sha256 if self.catalog_state else None,
            adapter_id=self.adapter.id,
            target_gfx=candidate.get("gfx"),
        )

    def restore_core(self, backup_path, *, apply=False):
        self._require_target()
        backup = load_backup(backup_path)
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
        if not isinstance(self.adapter, ExtensionProvider):
            raise CapabilityUnavailable(f"adapter {self.adapter.id} has no extension inventory capability")
        profiles = (self.catalog or {}).get("_comfyui_extension_profiles", {})
        extension_catalog = (self.catalog or {}).get("_extension_catalog", {})
        return self.adapter.extension_inventory(self.target, candidate, profiles, extension_catalog)

    def extension_resolve(self, candidate, selections=()):
        self._require_target()
        plan = self.extension_plan(candidate, selections)
        return run_extension_resolver(self.target, candidate, plan.extensions)

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
        backup = self.adapter.create_extension_backup(self.target, plan, backup_dir)
        return self.adapter.apply_extension_plan(self.target, plan, backup)

    def restore_extensions(self, backup_path, *, apply=False):
        self._require_target()
        backup = load_extension_backup(backup_path)
        if apply:
            return apply_extension_restore(self.target, backup)
        return InstallResult(plan=build_extension_restore_plan(self.target, backup))

    def _require_target(self):
        if self.target is None:
            raise ValueError("detect a target before running this operation")
