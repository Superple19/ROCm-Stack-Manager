"""ComfyUI implementation of the shared ROCm adapter contract."""

from ...core.adapter import CapabilityUnavailable
from ...core.backup import create_extension_backup, load_extension_backup
from ...core.inventory import collect_inventory
from ...core.install import apply_extension_restore, build_extension_restore_plan, build_install_plan
from ...core.hardware import probe_hardware
from ...core.extension_verification import build_extension_verification
from ...core.launch import LaunchOptions
from ...core.verify import probe_target, target_abi_tags, target_platform_tags, target_python_tag
from .detect import detect_comfyui
from .extensions import (
    PROFILES,
    apply_extension_plan,
    build_extension_plan,
    build_extension_report,
    package_names_for_extensions,
)


class ComfyUIAdapter:
    """Application adapter for ComfyUI portable and source layouts."""

    id = "comfyui"

    def detect(self, path="."):
        return detect_comfyui(path)

    def inventory(self, target, candidate=None):
        return collect_inventory(target, candidate)

    def plan(self, target, candidate, selections=(), **binding):
        if selections:
            raise CapabilityUnavailable(
                "ComfyUI core plans do not accept extensions; use extensions plan"
            )
        return build_install_plan(target, candidate, **binding)

    def verify(self, target):
        return probe_target(target)

    def hardware(self, target):
        return probe_hardware(target)

    def python_tag(self, target):
        return target_python_tag(target)

    @staticmethod
    def _candidate_for_target(target, candidate):
        if candidate is None:
            return None
        values = dict(candidate)
        platform_tags = target_platform_tags(target)
        if platform_tags:
            values["platform_tags"] = list(platform_tags)
        abi_tags = target_abi_tags(target)
        if abi_tags:
            values["abi_tags"] = list(abi_tags)
        return values

    def launch(self, target, options: LaunchOptions):
        raise CapabilityUnavailable("ComfyUI launch planning is not implemented")

    def extension_inventory(self, target, candidate=None, profile_documents=None, extension_catalog=None):
        inventory = self.inventory(target, candidate)
        candidate = self._candidate_for_target(target, candidate)
        return build_extension_report(inventory, profile_documents, candidate, extension_catalog)

    def extension_plan(
        self,
        target,
        candidate,
        selections=(),
        profile_documents=None,
        extension_catalog=None,
        allow_unverified=False,
    ):
        inventory = self.inventory(target, candidate)
        candidate = self._candidate_for_target(target, candidate)
        return build_extension_plan(
            target,
            inventory,
            candidate,
            profile_documents,
            selections,
            extension_catalog,
            allow_unverified,
        )

    def extension_verify(self, target, candidate, selections=(), profile_documents=None, extension_catalog=None):
        candidate = self._candidate_for_target(target, candidate)
        plan = self.extension_plan(target, candidate, selections, profile_documents, extension_catalog)
        runtime = self.verify(target)
        hardware = self.hardware(target)
        import_names = {profile.id: profile.import_names for profile in PROFILES}
        return build_extension_verification(
            target,
            candidate,
            plan.extensions,
            import_names,
            runtime,
            hardware,
        )

    def create_extension_backup(self, target, plan, destination=None):
        if not plan.selections:
            raise CapabilityUnavailable("extension apply requires explicit extension selections")
        blocked = [
            item
            for item in plan.extensions
            if item["status"] != "installable"
            and not (
                plan.allow_unverified
                and item["status"] == "unverified"
                and item.get("preflight_eligible")
            )
        ]
        if blocked:
            names = ", ".join(item["id"] for item in blocked)
            raise CapabilityUnavailable(
                f"extension plan contains non-installable selections: {names}"
            )
        extension_ids = tuple(item["id"] for item in plan.extensions)
        return create_extension_backup(
            target,
            package_names_for_extensions(extension_ids),
            extension_ids=extension_ids,
            candidate_id=plan.candidate.get("id"),
            destination=destination,
        )

    def apply_extension_plan(self, target, plan, backup):
        return apply_extension_plan(target, plan, backup)

    def restore_extensions(self, target, backup_path, apply=False):
        backup = load_extension_backup(backup_path, materialize=apply)
        if apply:
            return apply_extension_restore(target, backup)
        return build_extension_restore_plan(target, backup)
