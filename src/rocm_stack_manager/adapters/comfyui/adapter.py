"""ComfyUI implementation of the shared ROCm adapter contract."""

from ...core.adapter import CapabilityUnavailable
from ...core.inventory import collect_inventory
from ...core.launch import LaunchOptions, LaunchPlan
from ...core.planning import build_plan
from ...core.verify import probe_target, target_python_tag
from .detect import detect_comfyui
from .extensions import build_extension_report


class ComfyUIAdapter:
    """Application adapter for ComfyUI portable and source layouts."""

    id = "comfyui"

    def detect(self, path="."):
        return detect_comfyui(path)

    def inventory(self, target, candidate=None):
        return collect_inventory(target, candidate)

    def plan(self, target, candidate, selections=()):
        if selections:
            raise CapabilityUnavailable(
                "ComfyUI core plans do not accept extensions; use extensions plan"
            )
        return build_plan(target, candidate)

    def verify(self, target):
        return probe_target(target)

    def python_tag(self, target):
        return target_python_tag(target)

    def launch(self, target, options: LaunchOptions):
        raise CapabilityUnavailable("ComfyUI launch planning is not implemented")

    def extension_inventory(self, target, candidate=None, profile_documents=None):
        inventory = self.inventory(target, candidate)
        return build_extension_report(inventory, profile_documents, candidate)

    def extension_plan(self, target, candidate, selections=(), profile_documents=None):
        raise CapabilityUnavailable("ComfyUI extension planning is not implemented")
