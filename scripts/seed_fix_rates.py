#!/usr/bin/env python3
"""
scripts/seed_fix_rates.py
─────────────────────────
Populate (or refresh) ``data/fix_success_rates.json`` from the RCA staging
archive (``data/rca_staging_archive.json``).

For each fix candidate in the archive:
  key     = "{fix_type}:{error_type}"
  success = is_applicable AND NOT has_blockers  (1 = success, 0 = not)
  total   = 1

Merges with any existing entries so real human-submitted feedback is not lost.

Usage
─────
    python scripts/seed_fix_rates.py
    python scripts/seed_fix_rates.py --dry-run      # preview only, no write
    python scripts/seed_fix_rates.py --force         # overwrite even if already seeded
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR   = _SCRIPT_DIR.parent
_DATA_DIR   = _ROOT_DIR / "data"

ARCHIVE_FILE   = _DATA_DIR / "rca_staging_archive.json"
FIX_RATES_FILE = _DATA_DIR / "fix_success_rates.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_archive() -> list[dict]:
    if not ARCHIVE_FILE.exists():
        print(f"[ERROR] Archive not found: {ARCHIVE_FILE}")
        return []
    try:
        with ARCHIVE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to load archive: {exc}")
        return []


def load_existing_rates() -> dict:
    if not FIX_RATES_FILE.exists():
        return {}
    try:
        with FIX_RATES_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def seed_from_archive(
    archive: list[dict],
    existing: dict,
    force: bool = False,
) -> dict:
    """
    Build a merged fix_success_rates dict from archive + existing data.

    Parameters
    ----------
    archive  : list of archive entry dicts
    existing : current fix_success_rates dict (may be empty)
    force    : if True, reseed even if _metadata.archive_entry_count matches

    Returns
    -------
    dict ready to write to fix_success_rates.json
    """
    meta = existing.get("_metadata") or {}
    if not force and int(meta.get("archive_entry_count", 0)) >= len(archive):
        print(f"[INFO] Already up-to-date ({len(archive)} archive entries). "
              "Use --force to reseed.")
        return existing

    # Aggregate from archive
    agg: dict[str, dict] = defaultdict(lambda: {"success": 0, "total": 0})
    for entry in archive:
        rca        = entry.get("rca") or {}
        error_type = str(rca.get("error_type") or "unknown")
        fixes_list = (rca.get("fixes") or {}).get("fixes") or []
        for fix in fixes_list:
            fix_type = str(fix.get("fix_type") or "command")
            is_ok    = bool(fix.get("is_applicable")) and not bool(fix.get("has_blockers"))
            key      = f"{fix_type}:{error_type}"
            agg[key]["success"] += int(is_ok)
            agg[key]["total"]   += 1

    now_iso = _utc_now()
    merged: dict = {}

    # Merge archive data with existing (don't overwrite real feedback)
    for key, vals in agg.items():
        if key in existing and not key.startswith("_"):
            ex = existing[key]
            merged[key] = {
                "success":      ex.get("success", 0) + vals["success"],
                "total":        ex.get("total",   0) + vals["total"],
                "last_updated": now_iso,
            }
        else:
            merged[key] = {
                "success":      vals["success"],
                "total":        vals["total"],
                "last_updated": now_iso,
            }

    # Carry forward any real keys not covered by archive
    for key, val in existing.items():
        if key.startswith("_"):
            continue
        if key not in merged:
            merged[key] = val

    merged["_metadata"] = {
        "source":              "rca_staging_archive",
        "seeded_at":           now_iso,
        "archive_entry_count": len(archive),
    }
    return merged


def print_summary(merged: dict) -> None:
    print("\n--- fix_success_rates.json summary ---")
    keys = [k for k in merged if not k.startswith("_")]
    print(f"  Total entries : {len(keys)}")
    col_w = max((len(k) for k in keys), default=10) + 2
    print(f"  {'Key':<{col_w}}  {'Success':>7}  {'Total':>5}  {'Rate':>6}")
    print(f"  {'-'*col_w}  {'-------':>7}  {'-----':>5}  {'------':>6}")
    for key in sorted(keys):
        v = merged[key]
        s = int(v.get("success", 0))
        t = int(v.get("total",   0))
        r = f"{s/t*100:.1f}%" if t else "  N/A"
        print(f"  {key:<{col_w}}  {s:>7}  {t:>5}  {r:>6}")
    print()


def write_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(path)
    print(f"[OK] Written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed fix_success_rates.json from archive.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only; do not write file.")
    parser.add_argument("--force", action="store_true",
                        help="Reseed even if already up-to-date.")
    args = parser.parse_args()

    archive  = load_archive()
    if not archive:
        print("[WARN] Archive is empty — nothing to seed.")
        sys.exit(0)

    print(f"[INFO] Loaded {len(archive)} archive entries.")
    existing = load_existing_rates()
    print(f"[INFO] Existing fix_success_rates: {len([k for k in existing if not k.startswith('_')])} entries.")

    merged = seed_from_archive(archive, existing, force=args.force)
    print_summary(merged)

    if args.dry_run:
        print("[DRY-RUN] No file written.")
    else:
        write_file(FIX_RATES_FILE, merged)


if __name__ == "__main__":
    main()
