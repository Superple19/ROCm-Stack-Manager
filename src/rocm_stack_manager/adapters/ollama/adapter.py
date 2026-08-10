"""Ollama adapter scaffold."""

from ...core.adapter import CapabilityUnavailable
from ...core.hardware import probe_hardware
from ...core.launch import LaunchOptions, LaunchPlan


class OllamaAdapter:
    """Reserved adapter boundary for the native Ollama runtime."""

    id = "ollama"

    def _unimplemented(self):
        raise CapabilityUnavailable("Ollama adapter capabilities are not implemented yet")

    def detect(self, path="."):
        self._unimplemented()

    def inventory(self, target, candidate=None):
        self._unimplemented()

    def plan(self, target, candidate, selections=()):
        self._unimplemented()

    def verify(self, target):
        self._unimplemented()

    def hardware(self, target):
        return probe_hardware(target)

    def launch(self, target, options: LaunchOptions) -> LaunchPlan:
        self._unimplemented()
