import os
from pathlib import Path


def configure_runtime():
    """Point cache/config directories at writable workspace paths."""
    cache_root = Path(os.getenv("GRIDLOCK_CACHE_DIR", "artifacts/cache")).resolve()
    yolo_dir = cache_root / "ultralytics"
    mpl_dir = cache_root / "matplotlib"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    return {
        "cache_root": cache_root,
        "yolo_dir": yolo_dir,
        "mpl_dir": mpl_dir,
    }

