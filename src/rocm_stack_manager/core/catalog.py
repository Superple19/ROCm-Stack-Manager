"""Read versioned ROCm Evidence Matrix catalog snapshots."""

import json
from pathlib import Path


def load_catalog(path):
    """Load a local catalog snapshot without changing it."""

    catalog_path = Path(path).expanduser().resolve()
    with catalog_path.open(encoding="utf-8") as handle:
        return json.load(handle)
