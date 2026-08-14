"""Contracts shared by application-specific ROCm adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .launch import LaunchOptions, LaunchPlan


class AdapterError(RuntimeError):
    """Base error for adapter discovery and capability failures."""


class CapabilityUnavailable(AdapterError):
    """Raised when an adapter does not implement a requested capability."""


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Common target operations exposed by every application adapter."""

    id: str

    def detect(self, path: Path | str = ".") -> Any:
        ...

    def inventory(self, target: Any, candidate: dict | None = None) -> Any:
        ...

    def plan(
        self,
        target: Any,
        candidate: dict,
        selections: tuple[str, ...] = (),
        **binding: Any,
    ) -> Any:
        ...

    def verify(self, target: Any) -> Any:
        ...

    def launch(self, target: Any, options: LaunchOptions) -> LaunchPlan:
        ...


@runtime_checkable
class PythonPackageAdapter(Protocol):
    """Optional capability for adapters with a target Python resolver."""

    id: str

    def python_tag(self, target: Any) -> str | None:
        ...


@runtime_checkable
class TargetProbeAdapter(Protocol):
    """Optional capability for target-local runtime and GFX probing."""

    id: str

    def verify(self, target: Any) -> Any:
        ...


@runtime_checkable
class HardwareProvider(Protocol):
    """Optional capability for application-aware hardware detection."""

    def hardware(self, target: Any) -> Any:
        ...


@runtime_checkable
class ExtensionProvider(Protocol):
    """Optional application-specific extension inventory and planning."""

    id: str

    def extension_inventory(
        self,
        target: Any,
        candidate: dict | None = None,
        profile_documents: dict | None = None,
        extension_catalog: dict | None = None,
        inventory: Any | None = None,
    ) -> dict:
        ...

    def extension_plan(
        self,
        target: Any,
        candidate: dict,
        selections: tuple[str, ...] = (),
        profile_documents: dict | None = None,
        extension_catalog: dict | None = None,
        allow_unverified: bool = False,
    ) -> Any:
        ...


@runtime_checkable
class ExtensionInstaller(Protocol):
    """Optional capability for explicit extension apply and recovery."""

    id: str

    def create_extension_backup(self, target: Any, plan: Any, destination=None) -> Any:
        ...

    def apply_extension_plan(self, target: Any, plan: Any, backup: Any) -> Any:
        ...

    def restore_extensions(self, target: Any, backup_path, apply=False) -> Any:
        ...


@runtime_checkable
class ExtensionVerifier(Protocol):
    """Optional capability for target-local extension verification export."""

    id: str

    def extension_verify(
        self,
        target: Any,
        candidate: dict,
        selections: tuple[str, ...] = (),
        profile_documents: dict | None = None,
        extension_catalog: dict | None = None,
    ) -> dict:
        ...
