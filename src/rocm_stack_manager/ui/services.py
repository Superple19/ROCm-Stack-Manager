"""Presentation-facing services for the optional Manager UI."""

from dataclasses import dataclass
from pathlib import Path

from ..adapters.registry import get_adapter
from ..core.adapter import ExtensionInstaller, ExtensionProvider, PythonPackageAdapter
from ..core.backup import create_backup, load_backup, load_extension_backup
from ..core.catalog import ensure_catalog, iter_candidates, load_catalog
from ..core.install import (
    InstallResult,
    apply_extension_restore,
    apply_install,
    apply_restore,
    build_extension_restore_plan,
    build_restore_plan,
    dry_run_install,
)


@dataclass(frozen=True)
class CatalogState:
    path: Path
    source: str
    refreshed: bool


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
        self.catalog_state = CatalogState(
            path=catalog_path,
            source=source,
            refreshed=refresh,
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
        )

    def inventory(self, candidate=None):
        self._require_target()
        return self.adapter.inventory(self.target, candidate)

    def verify(self):
        self._require_target()
        return self.adapter.verify(self.target)

    def plan(self, candidate):
        self._require_target()
        return dry_run_install(self.target, candidate)

    def apply_core(self, candidate, *, allow_unverified=False, backup_dir=None):
        self._require_target()
        backup = create_backup(self.target, backup_dir)
        return apply_install(
            self.target,
            candidate,
            backup,
            allow_unverified=allow_unverified,
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
        return self.adapter.extension_plan(
            self.target,
            candidate,
            tuple(selections),
            profiles,
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
