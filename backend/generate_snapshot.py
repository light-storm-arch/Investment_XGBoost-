"""Generate snapshot data files for faster app startup.

Run this script periodically (e.g. quarterly) to update the bundled
historical data. Then commit the updated files in backend/data/.

Usage:
    cd backend && python generate_snapshot.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import SNAPSHOT_DIR, DATA_START_DATE
from data_fetcher import clear_cache

# Import the raw download functions — we need full history, not cached/snapshot data
import data_fetcher


def main():
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    # Clear cache so we fetch fresh full-history data
    clear_cache()

    # Temporarily disable snapshot loading by removing meta file if it exists
    meta_path = SNAPSHOT_DIR / "snapshot_meta.json"
    had_meta = meta_path.exists()
    if had_meta:
        old_meta = meta_path.read_text()
        meta_path.unlink()

    try:
        print(f"Fetching full history from {DATA_START_DATE}...")

        print("  Downloading fund prices...")
        fund = data_fetcher.fetch_fund_prices()
        fund.to_parquet(SNAPSHOT_DIR / "fund_prices_snapshot.parquet")
        print(f"  -> {len(fund)} rows saved")

        print("  Downloading FRED data...")
        fred = data_fetcher.fetch_fred_data()
        fred.to_parquet(SNAPSHOT_DIR / "fred_data_snapshot.parquet")
        print(f"  -> {len(fred)} rows saved")

        print("  Downloading auxiliary prices...")
        aux = data_fetcher.fetch_aux_prices()
        aux.to_parquet(SNAPSHOT_DIR / "aux_prices_snapshot.parquet")
        print(f"  -> {len(aux)} rows saved")

        cutoff = datetime.now().strftime("%Y-%m-%d")
        meta = {
            "cutoff_date": cutoff,
            "created_at": datetime.now().isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\nSnapshot saved to {SNAPSHOT_DIR}/")
        print(f"Cutoff date: {cutoff}")
        print("Commit these files to speed up app startup.")

    except Exception:
        # Restore old meta if we removed it
        if had_meta:
            meta_path.write_text(old_meta)
        raise

    # Clear cache again so next app run uses the fresh snapshot
    clear_cache()


if __name__ == "__main__":
    main()
