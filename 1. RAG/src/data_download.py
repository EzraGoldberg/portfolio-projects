# Reusable data ingestion utilities for downloading Kaggle datasets into the project's data/1_source/ directory.
"""
Usage (standalone):
    python src/data_download.py

Usage (import):
    from src.data_download import download_dataset, download_all, DATASETS
"""

# ---------------------------------------------------------------------------
# Imports
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import kagglehub
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# Project root is two levels above this file (project/src/data_download.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "1_source"

DATASETS: dict[str, str] = {
    "lgbt_EU":             "ruslankl/european-union-lgbt-survey-2012",
    "retractions":         "kanchana1990/global-scientific-retractions-19272026",
    "HIV_AIDS_data":       "imdevskp/hiv-aids-dataset",
    "UNICEF_Immunization": "fahimvj/immunization-data-unicef",
}

# ---------------------------------------------------------------------------
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment
def load_kaggle_credentials(env_path: Optional[Path] = None) -> None:
    """Load KAGGLE_USERNAME and KAGGLE_KEY from a .env file.

    kagglehub reads these directly from the environment, so we only need to
    ensure they are present before the first API call.

    Args:
        env_path: Path to the .env file.  Defaults to ``<project_root>/.env``.

    Raises:
        EnvironmentError: If either credential is missing after loading.
    """
    dotenv_file = env_path or PROJECT_ROOT / ".env"
    loaded = load_dotenv(dotenv_file)

    if loaded:
        log.debug("Loaded environment variables from %s", dotenv_file)
    else:
        log.debug(".env file not found at %s — relying on existing environment", dotenv_file)

    missing = [k for k in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing Kaggle credentials in environment: {missing}. "
            "Add them to your .env file or export them before running."
        )


# ---------------------------------------------------------------------------
# Core download logic
def download_dataset(
    dataset_id: str,
    target_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Download a single Kaggle dataset and copy it to *target_dir*.

    If *target_dir* already exists and is non-empty, the download is skipped
    unless *force* is ``True``.

    Args:
        dataset_id: Kaggle dataset handle in ``owner/dataset-slug`` format.
        target_dir:  Destination directory inside ``data/1_source/``.
        force:       Re-download and overwrite even if the directory exists.

    Returns:
        The resolved *target_dir* path.

    Raises:
        RuntimeError: If kagglehub fails to return a valid path.
    """
    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        log.info("⏭  Skipping %-40s — already exists at %s", dataset_id, target_dir)
        return target_dir

    log.info("⬇  Downloading %-40s …", dataset_id)

    # kagglehub downloads to its own cache directory; we then copy from there.
    cached_path_str: str = kagglehub.dataset_download(dataset_id)
    if not cached_path_str:
        raise RuntimeError(f"kagglehub returned an empty path for dataset '{dataset_id}'")

    cached_path = Path(cached_path_str)
    log.debug("   kagglehub cache path: %s", cached_path)

    # Wipe the destination first when forcing, otherwise create fresh.
    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
        log.debug("   Removed existing directory: %s", target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy every file/subdirectory from the cache into target_dir.
    for item in cached_path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    file_count = sum(1 for _ in target_dir.rglob("*") if _.is_file())
    log.info("✅  Saved %d file(s) to %s", file_count, target_dir)
    return target_dir


def download_all(
    datasets: Optional[dict[str, str]] = None,
    source_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Download every dataset in *datasets*.

    Args:
        datasets:   Mapping of ``name → kaggle_dataset_id``.
                    Defaults to the module-level :data:`DATASETS` dict.
        source_dir: Root destination directory.
                    Defaults to :data:`SOURCE_DIR` (``data/1_source/``).
        force:      Pass ``True`` to re-download even if a dataset exists.

    Returns:
        Mapping of ``name → resolved target path`` for each downloaded dataset.
    """
    datasets = datasets or DATASETS
    source_dir = source_dir or SOURCE_DIR
    source_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting download of %d dataset(s) into %s", len(datasets), source_dir)
    results: dict[str, Path] = {}

    for name, dataset_id in datasets.items():
        target = source_dir / name
        try:
            results[name] = download_dataset(dataset_id, target, force=force)
        except Exception as exc:  # noqa: BLE001
            log.error("❌  Failed to download '%s' (%s): %s", name, dataset_id, exc)

    log.info("Done. %d/%d dataset(s) available.", len(results), len(datasets))
    return results


# ---------------------------------------------------------------------------
# Inspection helpers
def list_dataset_files(name: str, source_dir: Optional[Path] = None) -> list[Path]:
    """Return a sorted list of all files inside a downloaded dataset directory.

    Args:
        name:       Dataset key as used in :data:`DATASETS`.
        source_dir: Override the default :data:`SOURCE_DIR`.

    Returns:
        Sorted list of :class:`~pathlib.Path` objects.
    """
    base = (source_dir or SOURCE_DIR) / name
    if not base.exists():
        log.warning("Dataset directory not found: %s", base)
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def verify_downloads(
    datasets: Optional[dict[str, str]] = None,
    source_dir: Optional[Path] = None,
) -> dict[str, bool]:
    """Check which datasets have been downloaded and are non-empty.

    Returns:
        Mapping of ``name → True/False``.
    """
    datasets = datasets or DATASETS
    source_dir = source_dir or SOURCE_DIR
    status: dict[str, bool] = {}

    for name in datasets:
        target = source_dir / name
        present = target.exists() and any(target.rglob("*"))
        status[name] = present
        icon = "✅" if present else "❌"
        log.info("%s  %-30s  %s", icon, name, target if present else "not found")

    return status

# def get_list_datasets(  # Made first, too noisy. Needed simpler version. Will remove old code on next commit
#     datasets: Optional[dict[str, str]] = None,
#     source_dir: Optional[Path] = None,
# ) -> list[str]:
#     """Return a list of dataset names that are downloaded and non-empty."""
#     status = verify_downloads(datasets=datasets, source_dir=source_dir)
#     return [name for name, present in status.items() if present]
def list_available_datasets(
    datasets: Optional[dict[str, str]] = None,
    source_dir: Optional[Path] = None,
) -> list[str]:
    datasets = datasets or DATASETS
    source_dir = source_dir or SOURCE_DIR

    return [
        name
        for name in datasets
        if (source_dir / name).exists() and any((source_dir / name).rglob("*"))
    ]

if __name__ == "__main__":
    load_kaggle_credentials()
    download_all()