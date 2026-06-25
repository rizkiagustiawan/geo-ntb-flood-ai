"""Data Retention Policy for A.E.C.O.

Manages disk space by cleaning up old satellite data, predictions,
and reports based on configurable retention periods.

Run periodically via cron or Celery beat.
"""

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Retention periods (days)
RETENTION_RAW = 7          # Raw satellite ZIPs
RETENTION_PREDICTIONS = 30 # Prediction GeoTIFFs
RETENTION_REPORTS = 14     # PDF reports
RETENTION_PATCHES = 90     # Training patches
RETENTION_HISTORY = 365    # Historical flood maps


def cleanup_old_files(directory: Path, pattern: str, max_age_days: int) -> int:
    """Delete files matching pattern older than max_age_days.

    Returns number of files deleted.
    """
    if not directory.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    deleted = 0

    for f in directory.glob(pattern):
        if not f.is_file():
            continue
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                f.unlink()
                deleted += 1
        except OSError as e:
            logger.warning("Could not delete %s: %s", f, e)

    return deleted


def run_retention_policy(dry_run: bool = False) -> dict:
    """Execute data retention policy.

    Parameters
    ----------
    dry_run : bool — If True, only report what would be deleted

    Returns
    -------
    dict — Summary of cleanup actions
    """
    logger.info("=" * 60)
    logger.info("DATA RETENTION POLICY (dry_run=%s)", dry_run)
    logger.info("=" * 60)

    actions = []

    # 1. Raw satellite ZIPs
    raw_dir = OUTPUTS_DIR / "predictions"
    count = cleanup_old_files(raw_dir, "*.zip", RETENTION_RAW)
    actions.append({"path": str(raw_dir), "pattern": "*.zip", "deleted": count,
                    "retention_days": RETENTION_RAW})
    logger.info("Raw ZIPs: %d deleted (retention: %d days)", count, RETENTION_RAW)

    # 2. Old predictions (keep final_flood_map.tif)
    pred_dir = OUTPUTS_DIR / "predictions"
    if pred_dir.exists():
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_PREDICTIONS)
        count = 0
        for f in pred_dir.glob("flood_mask_*.tif"):
            if not f.is_file():
                continue
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff and not dry_run:
                f.unlink()
                count += 1
        actions.append({"path": str(pred_dir), "pattern": "flood_mask_*.tif",
                        "deleted": count, "retention_days": RETENTION_PREDICTIONS})
        logger.info("Old predictions: %d deleted (retention: %d days)", count, RETENTION_PREDICTIONS)

    # 3. PDF reports
    reports_dir = Path("/tmp/reports")
    count = cleanup_old_files(reports_dir, "*.pdf", RETENTION_REPORTS)
    actions.append({"path": str(reports_dir), "pattern": "*.pdf", "deleted": count,
                    "retention_days": RETENTION_REPORTS})
    logger.info("PDF reports: %d deleted (retention: %d days)", count, RETENTION_REPORTS)

    # 4. Training patches (older ones)
    patches_dir = DATA_DIR / "patches"
    count = cleanup_old_files(patches_dir / "features", "*.npy", RETENTION_PATCHES)
    count += cleanup_old_files(patches_dir / "labels", "*.npy", RETENTION_PATCHES)
    actions.append({"path": str(patches_dir), "pattern": "*.npy", "deleted": count,
                    "retention_days": RETENTION_PATCHES})
    logger.info("Training patches: %d deleted (retention: %d days)", count, RETENTION_PATCHES)

    # 5. Historical flood maps
    history_dir = OUTPUTS_DIR / "history"
    count = cleanup_old_files(history_dir, "*.tif", RETENTION_HISTORY)
    actions.append({"path": str(history_dir), "pattern": "*.tif", "deleted": count,
                    "retention_days": RETENTION_HISTORY})
    logger.info("Historical maps: %d deleted (retention: %d days)", count, RETENTION_HISTORY)

    total_deleted = sum(a["deleted"] for a in actions)
    logger.info("Total files cleaned: %d", total_deleted)
    logger.info("=" * 60)

    return {"actions": actions, "total_deleted": total_deleted}


def get_disk_usage() -> dict:
    """Report disk usage for data and output directories."""
    dirs = {
        "data/raw": DATA_DIR / "raw",
        "data/processed": DATA_DIR / "processed",
        "data/patches": DATA_DIR / "patches",
        "outputs/models": OUTPUTS_DIR / "models",
        "outputs/predictions": OUTPUTS_DIR / "predictions",
        "outputs/history": OUTPUTS_DIR / "history",
    }

    usage = {}
    for name, path in dirs.items():
        if path.exists():
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            usage[name] = {
                "bytes": total,
                "mb": round(total / (1024 * 1024), 2),
                "files": sum(1 for f in path.rglob("*") if f.is_file()),
            }

    return usage


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Retention Policy")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't delete")
    parser.add_argument("--disk-usage", action="store_true", help="Show disk usage")
    args = parser.parse_args()

    if args.disk_usage:
        usage = get_disk_usage()
        for name, info in usage.items():
            print(f"  {name}: {info['mb']} MB ({info['files']} files)")
    else:
        run_retention_policy(dry_run=args.dry_run)
