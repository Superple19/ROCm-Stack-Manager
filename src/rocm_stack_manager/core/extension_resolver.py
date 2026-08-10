"""Targeted, non-mutating dependency checks for selected extensions."""

from dataclasses import dataclass
import subprocess

from .verify import _clean_environment


@dataclass(frozen=True)
class ExtensionResolverResult:
    extension_id: str
    extension_candidate_id: str | None
    candidate_id: str
    candidate_hash: str | None
    status: str
    command: tuple[str, ...]
    returncode: int | None = None
    output: str = ""
    claim_status: str | None = None
    preflight: bool = False

    def as_dict(self):
        return {
            "extension_id": self.extension_id,
            "extension_candidate_id": self.extension_candidate_id,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "status": self.status,
            "command": list(self.command),
            "returncode": self.returncode,
            "output": self.output,
            "claim_status": self.claim_status,
            "verification_scope": "preflight" if self.preflight else "install_plan",
            "promotion": "none",
            "network_access": True,
            "installation_performed": False,
        }


def _requirements(candidate, extension):
    core = tuple(candidate.get("wheel_urls") or ()) or tuple(candidate.get("package_specs") or ())
    extension_sources = tuple(extension.get("sources") or ())
    return tuple(dict.fromkeys(core + extension_sources))


def build_extension_resolver_command(target, candidate, extension):
    """Build a pip dry-run command for one selected core/extension pair."""

    if not extension.get("extension_candidate_id"):
        raise ValueError(f"extension {extension.get('id', 'unknown')} has no exact candidate identity")
    if extension.get("status") != "installable" and not extension.get("preflight_eligible"):
        raise ValueError(f"extension {extension.get('id', 'unknown')} is not eligible for preflight")
    requirements = _requirements(candidate, extension)
    if not requirements:
        raise ValueError(f"extension {extension.get('id', 'unknown')} has no exact source")
    if target.python_executable is None:
        raise ValueError("target Python executable was not found")
    command = [
        str(target.python_executable),
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--no-input",
        "--disable-pip-version-check",
        "--report",
        "-",
    ]
    if candidate.get("index_url") and not candidate.get("wheel_urls"):
        command.extend(("--index-url", candidate["index_url"]))
    command.extend(requirements)
    return tuple(command)


def run_extension_resolver(target, candidate, extensions, *, timeout=900, runner=subprocess.run):
    """Resolve only the selected extension records; never install anything."""

    results = []
    for extension in extensions:
        try:
            command = build_extension_resolver_command(target, candidate, extension)
        except ValueError as error:
            results.append(
                ExtensionResolverResult(
                    extension_id=extension.get("id", "unknown"),
                    extension_candidate_id=extension.get("extension_candidate_id"),
                    candidate_id=candidate.get("id", "unknown"),
                    candidate_hash=candidate.get("candidate_hash"),
                    status="not_applicable",
                    command=(),
                    output=str(error),
                    claim_status=extension.get("claim_status"),
                    preflight=bool(extension.get("preflight_eligible")),
                )
            )
            continue
        try:
            completed = runner(
                list(command),
                cwd=str(target.comfyui_dir),
                env=_clean_environment(target),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            output = ((completed.stdout or "") + (completed.stderr or ""))[-12000:]
            status = "resolver_verified" if completed.returncode == 0 else "resolver_failed"
            results.append(
                ExtensionResolverResult(
                    extension_id=extension.get("id", "unknown"),
                    extension_candidate_id=extension.get("extension_candidate_id"),
                    candidate_id=candidate.get("id", "unknown"),
                    candidate_hash=candidate.get("candidate_hash"),
                    status=status,
                    command=command,
                    returncode=completed.returncode,
                    output=output,
                    claim_status=extension.get("claim_status"),
                    preflight=bool(extension.get("preflight_eligible")),
                )
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            results.append(
                ExtensionResolverResult(
                    extension_id=extension.get("id", "unknown"),
                    extension_candidate_id=extension.get("extension_candidate_id"),
                    candidate_id=candidate.get("id", "unknown"),
                    candidate_hash=candidate.get("candidate_hash"),
                    status="resolver_failed",
                    command=command,
                    output=f"{type(error).__name__}: {error}",
                    claim_status=extension.get("claim_status"),
                    preflight=bool(extension.get("preflight_eligible")),
                )
            )
    return tuple(results)
