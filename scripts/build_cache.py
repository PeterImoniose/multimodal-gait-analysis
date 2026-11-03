"""Process every trial and write the analysis caches (data/cache, results/tables).

    python scripts/run_audit.py      # first: per-file quality tables
    python scripts/build_cache.py    # then: events, features, EMG cycles, EEG metrics
"""

import time

from mgait import pipeline

if __name__ == "__main__":
    t0 = time.time()
    pipeline.build()
    print(f"done in {time.time() - t0:.0f}s")
