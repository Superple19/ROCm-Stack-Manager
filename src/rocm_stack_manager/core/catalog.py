"""Read and filter local ROCm Evidence Matrix catalog snapshots."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from urllib.parse import unquote

from .profile import ProfileError, evaluate_candidate, load_profile


class CatalogError(ValueError):
    """Raised when a Matrix catalog cannot be loaded or interpreted."""


DEFAULT_MATRIX_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/Superple19/rocm-evidence-matrix/main"
)
_PROFILE_PATHS = (
    "profiles/comfyui/profile.json",
    "profiles/comfyui/extensions/bitsandbytes.json",
    "profiles/comfyui/extensions/flash-attention.json",
    "profiles/comfyui/extensions/aiter.json",
    "profiles/comfyui/extensions/sageattention.json",
    "profiles/comfyui/extensions/triton.json",
)


def default_catalog_cache_dir():
    """Return the per-user cache directory for the fetched Matrix snapshot."""

    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    else:
        root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(root) / "rocm-stack-manager" / "matrix"


def _download_bytes(url, timeout=30):
    request = Request(url, headers={"User-Agent": "rocm-stack-manager/catalog"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (OSError, HTTPError, URLError) as error:
        raise CatalogError(f"cannot fetch Matrix catalog source {url}: {error}") from error


def _write_bytes_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _safe_relative_path(value):
    if not isinstance(value, str) or not value:
        raise CatalogError("Matrix catalog contains an invalid artifact path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CatalogError(f"Matrix catalog contains an unsafe artifact path: {value}")
    return relative


def ensure_catalog(
    path=None,
    *,
    cache_dir=None,
    base_url=DEFAULT_MATRIX_RAW_BASE_URL,
    refresh=False,
):
    """Resolve an explicit catalog or fetch the official Matrix snapshot.

    An explicit path never performs network access. When omitted, the first
    user-triggered catalog command downloads the generated catalog, all files
    it references, and the ComfyUI profiles into a per-user cache.
    """

    if path is not None:
        return Path(path).expanduser().resolve()

    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else default_catalog_cache_dir()
    catalog_path = cache_root / "data" / "catalog.json"
    if catalog_path.is_file() and not refresh:
        return catalog_path

    base = str(base_url or DEFAULT_MATRIX_RAW_BASE_URL).rstrip("/")
    staging_parent = cache_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="matrix-", dir=str(staging_parent)))
    try:
        catalog_bytes = _download_bytes(f"{base}/data/catalog.json")
        try:
            catalog_document = json.loads(catalog_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CatalogError("downloaded Matrix catalog is not valid UTF-8 JSON") from error
        relative_paths = []
        for artifact in catalog_document.get("artifacts", []):
            if not isinstance(artifact, dict) or not artifact.get("path"):
                continue
            relative_paths.append(_safe_relative_path(artifact["path"]))
        required_paths = tuple(dict.fromkeys(relative_paths))
        for relative_path in required_paths:
            content = _download_bytes(f"{base}/{relative_path.as_posix()}")
            _write_bytes_atomic(staging / relative_path, content)
        for relative_path in _PROFILE_PATHS:
            try:
                content = _download_bytes(f"{base}/{relative_path}")
            except CatalogError:
                continue
            _write_bytes_atomic(staging / relative_path, content)
        _write_bytes_atomic(staging / "data" / "catalog.json", catalog_bytes)

        files = sorted(path for path in staging.rglob("*") if path.is_file())
        catalog_destination = cache_root / "data" / "catalog.json"
        for source in files:
            relative = source.relative_to(staging)
            if relative.as_posix() == "data/catalog.json":
                continue
            destination = cache_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        catalog_destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging / "data" / "catalog.json", catalog_destination)
        manifest = {
            "source": base,
            "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "artifact_count": len(required_paths),
        }
        _write_bytes_atomic(
            cache_root / "source-manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return catalog_destination
    except CatalogError:
        raise
    except OSError as error:
        raise CatalogError(f"cannot store Matrix catalog cache at {cache_root}: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise CatalogError(f"cannot read catalog: {path}") from error
    except json.JSONDecodeError as error:
        raise CatalogError(f"invalid JSON catalog: {path}") from error


def load_catalog(path):
    """Load a Matrix index or matrix snapshot from a local path.

    A Matrix ``data/catalog.json`` is an index of generated artifacts. The
    manager consumes its ``compatibility_matrix`` artifact automatically, so
    callers may also pass ``data/matrix.json`` directly.
    """

    catalog_path = Path(path).expanduser().resolve()
    document = _read_json(catalog_path)
    if "targets" in document:
        _attach_profile(document, catalog_path)
        return document

    artifacts = document.get("artifacts", [])
    matrix_artifact = next(
        (item for item in artifacts if item.get("id") == "compatibility_matrix"),
        None,
    )
    if not matrix_artifact or not matrix_artifact.get("path"):
        raise CatalogError("catalog does not reference a compatibility matrix")

    relative_path = Path(matrix_artifact["path"])
    path_candidates = [catalog_path.parent / relative_path]
    if catalog_path.parent.name == "data":
        path_candidates.append(catalog_path.parent.parent / relative_path)
    matrix_path = next((candidate.resolve() for candidate in path_candidates if candidate.is_file()), None)
    if matrix_path is None:
        raise CatalogError(f"compatibility matrix is missing: {path_candidates[-1].resolve()}")
    matrix = _read_json(matrix_path)
    if not isinstance(matrix.get("targets"), list):
        raise CatalogError(f"compatibility matrix has no targets: {matrix_path}")
    matrix["_package_snapshots"] = {}
    matrix["_historical_candidates"] = []
    _attach_profile(matrix, catalog_path)
    for artifact in artifacts:
        artifact_id = artifact.get("id", "")
        if not artifact.get("path"):
            continue
        artifact_path_candidates = [catalog_path.parent / Path(artifact["path"])]
        if catalog_path.parent.name == "data":
            artifact_path_candidates.append(catalog_path.parent.parent / Path(artifact["path"]))
        artifact_path = next(
            (candidate.resolve() for candidate in artifact_path_candidates if candidate.is_file()),
            None,
        )
        if artifact_path:
            document = _read_json(artifact_path)
            if artifact_id.startswith("package_snapshots:"):
                matrix["_package_snapshots"][artifact_id] = document
            elif artifact_id == "package_history":
                matrix["_historical_candidates"] = document.get("candidates", [])
    return matrix


def _attach_profile(catalog, catalog_path):
    roots = []
    if catalog_path.parent.name == "data":
        roots.append(catalog_path.parent.parent)
    roots.append(catalog_path.parent)
    for root in roots:
        profile_path = root / "profiles" / "comfyui" / "profile.json"
        if not profile_path.is_file():
            continue
        try:
            catalog["_comfyui_profile"] = load_profile(profile_path)
        except ProfileError as error:
            raise CatalogError(str(error)) from error
        catalog["_comfyui_profile_path"] = str(profile_path)
        extension_profiles = {}
        extension_root = profile_path.parent / "extensions"
        for extension_path in sorted(extension_root.glob("*.json")):
            document = _read_json(extension_path)
            if document.get("id", "").startswith("comfyui."):
                extension_profiles[document["id"].removeprefix("comfyui.")] = document
        catalog["_comfyui_extension_profiles"] = extension_profiles
        return


def _candidate_id(platform, channel, gfx, rocm_version, torch_version):
    values = (platform, channel, gfx, rocm_version or "unknown", torch_version or "unknown")
    return "therock:" + ":".join(str(value).replace(":", "_") for value in values)


def _python_compatibility(catalog, platform, channel, gfx, python_tag):
    if not python_tag:
        return "unknown"
    source_id = f"packages-{channel}" + ("-linux" if platform == "linux" else "")
    snapshot = catalog.get("_package_snapshots", {}).get(f"package_snapshots:{source_id.removeprefix('packages-')}")
    if not snapshot:
        return "unknown"
    package_names = ("torch", "torchvision", f"amd-torch-device-{gfx}", f"amd-torchvision-device-{gfx}", f"rocm-sdk-device-{gfx}")
    package_results = []
    platform_tag = "win_amd64" if platform == "windows" else "linux_x86_64"
    for package_name in package_names:
        artifacts = snapshot.get("packages", {}).get(package_name)
        if artifacts is None:
            continue
        if not artifacts:
            package_results.append(False)
            continue
        package_results.append(
            any(
                item.get("python_tag") in {python_tag, "py3", "source"}
                and item.get("platform_tag") in {platform_tag, "any", "source"}
                for item in artifacts
            )
        )
    if not package_results:
        return "unknown"
    return "compatible" if all(package_results) else "incompatible"


def _candidate_from_channel(target, platform, channel, details, catalog, python_tag):
    rocm_version = details.get("rocm_device_version")
    torch_version = details.get("torch_device_version")
    torchvision_version = details.get("torchvision_device_version")
    available = bool(details.get("all_device_packages_available"))
    source_id = details.get("source_id")
    source = (catalog.get("sources") or {}).get(source_id, {})
    package_specs = []
    for package_name, version in (
        ("rocm", rocm_version),
        ("torch", torch_version),
        ("torchvision", torchvision_version),
    ):
        if version:
            package_specs.append(f"{package_name}=={version}")
    if target["gfx"] and rocm_version and torch_version and torchvision_version:
        package_specs.extend(
            (
                f"rocm-sdk-device-{target['gfx']}=={rocm_version}",
                f"amd-torch-device-{target['gfx']}=={torch_version}",
                f"amd-torchvision-device-{target['gfx']}=={torchvision_version}",
            )
        )
    candidate = {
        "id": _candidate_id(platform, channel, target["gfx"], rocm_version, torch_version),
        "distribution_family": "therock",
        "platform": platform,
        "channel": channel,
        "gfx": target["gfx"],
        "rocm_version": rocm_version,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "source_id": source_id,
        "index_url": source.get("url"),
        "package_specs": package_specs,
        "candidate_kind": (
            "installable"
            if available and package_specs
            else "artifact_only"
            if available
            else "unavailable"
        ),
        "artifact_available": available,
        "status": "artifact_available" if available else "artifact_unavailable",
        "lifecycle": "current",
        "python_tag": python_tag,
        "python_compatibility": _python_compatibility(catalog, platform, channel, target["gfx"], python_tag),
    }
    profile_status, profile_warnings, profile_id = evaluate_candidate(
        candidate, catalog.get("_comfyui_profile")
    )
    candidate.update(
        {
            "profile_id": profile_id,
            "profile_status": profile_status,
            "profile_warnings": list(profile_warnings),
        }
    )
    return candidate


def _historical_candidate(candidate, gfx, python_tag, profile=None):
    gfx_targets = set(candidate.get("available_gfx_targets") or candidate.get("gfx_targets") or [])
    if gfx not in gfx_targets:
        return None
    python_tags = set(candidate.get("python_tags") or [])
    if not python_tag:
        python_compatibility = "unknown"
    elif not python_tags:
        python_compatibility = "unknown"
    else:
        python_compatibility = "compatible" if python_tag in python_tags else "incompatible"
    artifact_available = bool(candidate.get("artifact_available"))
    evidence_status = candidate.get("evidence_status") or {}
    wheel_urls = list(candidate.get("wheel_urls") or [])
    rocm_version = candidate.get("rocm_version")
    if candidate.get("distribution_family", "legacy") == "legacy" and rocm_version:
        marker = f"/rocm-rel-{rocm_version}/"
        source_url = next(
            (
                url.rsplit("/", 1)[0] + f"/rocm-{rocm_version}.tar.gz"
                for url in wheel_urls
                if marker in unquote(url)
            ),
            None,
        )
        if source_url and source_url not in wheel_urls:
            wheel_urls.insert(0, source_url)
    candidate_result = {
        "id": candidate.get("id"),
        "distribution_family": candidate.get("distribution_family", "legacy"),
        "platform": candidate.get("platform"),
        "channel": candidate.get("channel"),
        "gfx": gfx,
        "rocm_version": rocm_version,
        "torch_version": candidate.get("torch_version"),
        "torchvision_version": candidate.get("torchvision_version"),
        "torchaudio_version": candidate.get("torchaudio_version"),
        "source_id": candidate.get("source_id"),
        "artifact_available": artifact_available,
        "status": "artifact_available" if artifact_available else "artifact_unavailable",
        "lifecycle": candidate.get("lifecycle", "historical"),
        "python_tag": python_tag,
        "python_compatibility": python_compatibility,
        "resolver_status": evidence_status.get("resolver", "not_collected"),
        "wheel_urls": wheel_urls,
        "package_specs": [],
        "candidate_kind": "installable" if artifact_available and wheel_urls else (
            "artifact_only" if artifact_available else "unavailable"
        ),
    }
    profile_status, profile_warnings, profile_id = evaluate_candidate(candidate_result, profile)
    candidate_result.update(
        {
            "profile_id": profile_id,
            "profile_status": profile_status,
            "profile_warnings": list(profile_warnings),
        }
    )
    return candidate_result


def iter_candidates(
    catalog,
    *,
    platform,
    gfx=None,
    channel=None,
    rocm_version=None,
    python_tag=None,
    include_unavailable=False,
    include_incompatible=False,
):
    """Return install candidates for one platform and optional filters.

    The Matrix records package channels either under a platform or, in older
    snapshots, directly on a target. Both shapes are accepted. This function
    only reports artifact evidence; it never promotes a candidate to runtime or
    hardware compatibility.
    """

    candidates = []
    for target in catalog.get("targets", []):
        target_gfx = target.get("gfx")
        if gfx and target_gfx != gfx:
            continue
        platform_record = (target.get("platforms") or {}).get(platform)
        channels = (platform_record or {}).get("package_channels")
        if channels is None:
            channels = target.get("package_channels", {}) if platform == "windows" else {}
        for channel_name, details in channels.items():
            if channel and channel_name != channel:
                continue
            if rocm_version and (details or {}).get("rocm_device_version") != rocm_version:
                continue
            candidate = _candidate_from_channel(
                target, platform, channel_name, details or {}, catalog, python_tag
            )
            if (
                (include_unavailable or candidate["artifact_available"])
                and (include_incompatible or candidate["python_compatibility"] != "incompatible")
            ):
                candidates.append(candidate)
    for historical in catalog.get("_historical_candidates", []):
        if historical.get("platform") != platform:
            continue
        if channel and historical.get("channel") != channel:
            continue
        if rocm_version and historical.get("rocm_version") != rocm_version:
            continue
        if not gfx:
            continue
        candidate = _historical_candidate(
            historical, gfx, python_tag, catalog.get("_comfyui_profile")
        )
        if not candidate:
            continue
        if (
            (include_unavailable or candidate["artifact_available"])
            and (include_incompatible or candidate["python_compatibility"] != "incompatible")
        ):
            candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda item: (
            item["gfx"],
            0 if item.get("lifecycle") == "current" else 1,
            {"stable": 0, "nightly": 1, "staging": 2}.get(item["channel"], 9),
            item["rocm_version"] or "",
            item["torch_version"] or "",
        ),
    )


def find_candidate(catalog, candidate_id):
    """Find an exact candidate ID, including unavailable artifact records."""

    for platform in ("windows", "linux"):
        for candidate in iter_candidates(catalog, platform=platform, include_unavailable=True):
            if candidate["id"] == candidate_id:
                return candidate
    raise CatalogError(f"candidate not found: {candidate_id}")
