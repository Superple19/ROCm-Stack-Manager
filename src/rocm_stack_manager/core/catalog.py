"""Read and filter local ROCm Evidence Matrix catalog snapshots."""

import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from urllib.parse import unquote

from .profile import ProfileError, evaluate_candidate, load_profile
from .identity import candidate_hash


class CatalogError(ValueError):
    """Raised when a Matrix catalog cannot be loaded or interpreted."""


DEFAULT_MATRIX_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/Superple19/rocm-evidence-matrix/main"
)
MATRIX_CATALOG_URL_ENV = "ROCM_MATRIX_CATALOG_URL"
_SHA256_LENGTH = 64


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_catalog_index(document):
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise CatalogError("Matrix catalog has an unsupported schema version")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        raise CatalogError("Matrix catalog has no artifact list")
    seen = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not all(
            isinstance(artifact.get(field), str)
            for field in ("id", "path", "schema")
        ) or not isinstance(artifact.get("schema_version"), int):
            raise CatalogError("Matrix catalog contains an invalid artifact entry")
        if artifact["id"] in seen:
            raise CatalogError(f"Matrix catalog contains duplicate artifact: {artifact['id']}")
        seen.add(artifact["id"])
        digest = artifact.get("sha256")
        if not _is_sha256(digest):
            raise CatalogError(f"Matrix catalog artifact has no valid SHA-256: {artifact['id']}")
        _safe_relative_path(artifact["path"])


def _validate_matrix_document(document):
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise CatalogError("compatibility matrix has an unsupported schema version")
    if not isinstance(document.get("targets"), list):
        raise CatalogError("compatibility matrix has no targets")
    for target in document["targets"]:
        if not isinstance(target, dict) or not re.fullmatch(r"gfx[0-9a-z]+", str(target.get("gfx", ""))):
            raise CatalogError("compatibility matrix contains an invalid GFX target")
        platforms = target.get("platforms")
        if not isinstance(platforms, dict):
            raise CatalogError(f"compatibility matrix has invalid platforms for {target['gfx']}")
        for platform_record in platforms.values():
            if not isinstance(platform_record, dict):
                raise CatalogError(f"compatibility matrix has an invalid platform record for {target['gfx']}")
            channels = platform_record.get("package_channels", {})
            if not isinstance(channels, dict):
                raise CatalogError(f"compatibility matrix has invalid package channels for {target['gfx']}")
            for channel, details in channels.items():
                if not isinstance(details, dict) or not isinstance(details.get("source_id"), str):
                    raise CatalogError(f"compatibility matrix has invalid {channel} package evidence for {target['gfx']}")
                if "torchaudio_version" not in details:
                    raise CatalogError(f"compatibility matrix lacks TorchAudio evidence for {target['gfx']} / {channel}")
                for field in ("rocm_device_version", "torch_device_version", "torchvision_device_version", "torchaudio_version"):
                    if details[field] is not None and not isinstance(details[field], str):
                        raise CatalogError(f"compatibility matrix has invalid {field} for {target['gfx']} / {channel}")
                if "all_device_packages_available" in details and not isinstance(details["all_device_packages_available"], bool):
                    raise CatalogError(f"compatibility matrix has invalid availability for {target['gfx']} / {channel}")


def _cached_catalog_path(cache_root):
    pointer_path = cache_root / "current.json"
    candidates = []
    if pointer_path.is_file():
        try:
            pointer = _read_json(pointer_path)
            if pointer.get("schema_version") != 1:
                raise CatalogError("unsupported Matrix cache pointer")
            snapshot = _safe_relative_path(pointer.get("snapshot"))
            catalog_path = cache_root / snapshot / "data" / "catalog.json"
            expected_hash = pointer.get("catalog_sha256")
            if not _is_sha256(expected_hash):
                raise CatalogError("Matrix cache pointer has no catalog hash")
            if catalog_path.is_file() and hashlib.sha256(catalog_path.read_bytes()).hexdigest() == expected_hash:
                candidates.append(catalog_path)
        except CatalogError:
            pass
    candidates.append(cache_root / "data" / "catalog.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            _validate_cached_bundle(candidate)
        except CatalogError:
            continue
        return candidate
    return None


def _validate_cached_bundle(catalog_path):
    document = _read_json(catalog_path)
    if "targets" in document:
        _validate_matrix_document(document)
        return
    _validate_catalog_index(document)
    root = catalog_path.parent.parent if catalog_path.parent.name == "data" else catalog_path.parent
    for artifact in document["artifacts"]:
        artifact_path = (root / _safe_relative_path(artifact["path"])).resolve()
        try:
            artifact_path.relative_to(root.resolve())
        except ValueError as error:
            raise CatalogError(f"Matrix artifact escapes cache root: {artifact['path']}") from error
        if not artifact_path.is_file():
            raise CatalogError(f"Matrix artifact is missing: {artifact['path']}")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise CatalogError(f"Matrix artifact hash mismatch: {artifact['id']}")
        artifact_document = _read_json(artifact_path)
        if not isinstance(artifact_document, dict):
            raise CatalogError(f"Matrix artifact is not a JSON object: {artifact['id']}")
        if artifact_document.get("schema_version") != artifact["schema_version"]:
            raise CatalogError(f"Matrix artifact schema version mismatch: {artifact['id']}")
        if artifact["id"].startswith("package_snapshots:") and not isinstance(artifact_document.get("packages"), dict):
            raise CatalogError(f"Matrix package snapshot has no package map: {artifact['id']}")
        if artifact["id"] == "extension_catalog" and not isinstance(artifact_document.get("extensions"), list):
            raise CatalogError("Matrix extension catalog has no extension list")
    matrix_artifact = next((item for item in document["artifacts"] if item["id"] == "compatibility_matrix"), None)
    if matrix_artifact is None:
        raise CatalogError("catalog does not reference a compatibility matrix")
    _validate_matrix_document(_read_json(root / _safe_relative_path(matrix_artifact["path"])))


def default_catalog_cache_dir():
    """Return the per-user cache directory for the fetched Matrix snapshot."""

    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    else:
        root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(root) / "rocm-stack-manager" / "matrix"


def default_matrix_raw_base_url():
    """Return the configured Matrix source or the public default."""

    return os.environ.get(MATRIX_CATALOG_URL_ENV) or DEFAULT_MATRIX_RAW_BASE_URL


def _download_bytes(url, timeout=30):
    request = Request(url, headers={"User-Agent": "rocm-stack-manager/catalog"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (OSError, HTTPError, URLError) as error:
        raise CatalogError(
            f"cannot fetch Matrix catalog source {url}: {error}; "
            "use --catalog for an offline file or set ROCM_MATRIX_CATALOG_URL for a trusted mirror"
        ) from error


def _write_bytes_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise CatalogError("Matrix catalog contains an invalid artifact path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CatalogError(f"Matrix catalog contains an unsafe artifact path: {value}")
    return relative


def ensure_catalog(
    path=None,
    *,
    cache_dir=None,
    base_url=None,
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
    if not refresh:
        cached = _cached_catalog_path(cache_root)
        if cached is not None:
            return cached

    base = str(base_url or default_matrix_raw_base_url()).rstrip("/")
    staging_parent = cache_root / "snapshots"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="matrix-", dir=str(staging_parent)))
    try:
        catalog_bytes = _download_bytes(f"{base}/data/catalog.json")
        try:
            catalog_document = json.loads(catalog_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CatalogError("downloaded Matrix catalog is not valid UTF-8 JSON") from error
        _validate_catalog_index(catalog_document)
        relative_paths = []
        for artifact in catalog_document.get("artifacts", []):
            if not isinstance(artifact, dict) or not artifact.get("path"):
                continue
            relative_paths.append(_safe_relative_path(artifact["path"]))
        required_paths = tuple(dict.fromkeys(relative_paths))
        for relative_path in required_paths:
            content = _download_bytes(f"{base}/{relative_path.as_posix()}")
            artifact = next(item for item in catalog_document["artifacts"] if item["path"] == relative_path.as_posix())
            if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
                raise CatalogError(f"downloaded Matrix artifact hash mismatch: {artifact['id']}")
            _write_bytes_atomic(staging / relative_path, content)
        _write_bytes_atomic(staging / "data" / "catalog.json", catalog_bytes)
        catalog_destination = staging / "data" / "catalog.json"
        _validate_cached_bundle(catalog_destination)
        snapshot_name = hashlib.sha256(catalog_bytes).hexdigest()
        snapshot_destination = cache_root / "snapshots" / snapshot_name
        if snapshot_destination.exists():
            existing_catalog = snapshot_destination / "data" / "catalog.json"
            try:
                _validate_cached_bundle(existing_catalog)
            except CatalogError:
                shutil.rmtree(snapshot_destination)
                os.replace(staging, snapshot_destination)
            else:
                shutil.rmtree(staging, ignore_errors=True)
        else:
            os.replace(staging, snapshot_destination)
        catalog_destination = snapshot_destination / "data" / "catalog.json"
        manifest = {
            "source": base,
            "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "artifact_count": len(required_paths),
            "snapshot": f"snapshots/{snapshot_name}",
        }
        _write_bytes_atomic(
            cache_root / "source-manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        _write_bytes_atomic(
            cache_root / "current.json",
            (json.dumps({"schema_version": 1, "snapshot": f"snapshots/{snapshot_name}", "catalog_sha256": snapshot_name}, indent=2, sort_keys=True) + "\n").encode("utf-8"),
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
        _validate_matrix_document(document)
        _attach_profile(document, catalog_path)
        return document

    _validate_cached_bundle(catalog_path)
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
    matrix["_extension_catalog"] = {
        "schema_version": 1,
        "generated_at": matrix.get("generated_at"),
        "sources": {},
        "extensions": [],
    }
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
            elif artifact_id == "extension_catalog":
                matrix["_extension_catalog"] = document
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


def _candidate_id(platform, channel, gfx, rocm_version, torch_version, torchvision_version, torchaudio_version):
    values = (
        platform,
        channel,
        gfx,
        rocm_version or "unknown",
        torch_version or "unknown",
        torchvision_version or "unknown",
        torchaudio_version or "unknown",
    )
    return "therock:" + ":".join(str(value).replace(":", "_") for value in values)


def _python_compatibility(catalog, platform, channel, gfx, python_tag):
    if not python_tag:
        return "unknown"
    source_id = f"packages-{channel}" + ("-linux" if platform == "linux" else "")
    snapshot = catalog.get("_package_snapshots", {}).get(f"package_snapshots:{source_id.removeprefix('packages-')}")
    if not snapshot:
        return "unknown"
    package_names = (
        "torch",
        "torchvision",
        "torchaudio",
        f"amd-torch-device-{gfx}",
        f"amd-torchvision-device-{gfx}",
        f"rocm-sdk-device-{gfx}",
    )
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
    torchaudio_version = details.get("torchaudio_version")
    available = bool(details.get("all_device_packages_available"))
    source_id = details.get("source_id")
    source = (catalog.get("sources") or {}).get(source_id, {})
    package_specs = []
    for package_name, version in (
        ("rocm", rocm_version),
        ("torch", torch_version),
        ("torchvision", torchvision_version),
        ("torchaudio", torchaudio_version),
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
        "id": _candidate_id(
            platform,
            channel,
            target["gfx"],
            rocm_version,
            torch_version,
            torchvision_version,
            torchaudio_version,
        ),
        "distribution_family": "therock",
        "platform": platform,
        "channel": channel,
        "gfx": target["gfx"],
        "rocm_version": rocm_version,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "torchaudio_version": torchaudio_version,
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
    candidate["candidate_hash"] = candidate_hash(candidate)
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
    candidate_result["candidate_hash"] = candidate_hash(candidate_result)
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
    distribution_family=None,
    lifecycle=None,
    candidate_kind=None,
):
    """Return install candidates for one platform and optional filters.

    The Matrix records package channels either under a platform or, in older
    snapshots, directly on a target. Both shapes are accepted. This function
    only reports artifact evidence; it never promotes a candidate to runtime or
    hardware compatibility.
    """

    candidates = []

    def include(candidate):
        if distribution_family and candidate.get("distribution_family") != distribution_family:
            return False
        if lifecycle and candidate.get("lifecycle") != lifecycle:
            return False
        if candidate_kind and candidate.get("candidate_kind") != candidate_kind:
            return False
        return True

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
                and include(candidate)
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
            and include(candidate)
        ):
            candidates.append(candidate)

    def version_key(value):
        text = str(value or "").strip()
        if not text:
            return (1, 0, 0, 0, 0, 0, "")
        match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)$", text)
        if not match:
            return (0, 0, 0, 0, 0, 0, text)
        major, minor, patch, suffix = match.groups()
        date_match = re.search(r"(\d{8})", suffix)
        date = int(date_match.group(1)) if date_match else 0
        release = 1 if not suffix or suffix.startswith("+") else 0
        return (
            0,
            -int(major),
            -int(minor or 0),
            -int(patch or 0),
            -release,
            -date,
            suffix,
        )

    return sorted(
        candidates,
        key=lambda item: (
            item["gfx"],
            0 if item.get("lifecycle") == "current" else 1,
            {"stable": 0, "nightly": 1, "staging": 2}.get(item["channel"], 9),
            version_key(item.get("rocm_version")),
            version_key(item.get("torch_version")),
            item.get("id") or "",
        ),
    )


def find_candidate(catalog, candidate_id):
    """Find an exact candidate ID, including unavailable artifact records."""

    for platform in ("windows", "linux"):
        for candidate in iter_candidates(catalog, platform=platform, include_unavailable=True):
            if candidate["id"] == candidate_id:
                return candidate
    raise CatalogError(f"candidate not found: {candidate_id}")
