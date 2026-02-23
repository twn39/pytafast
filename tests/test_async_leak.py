import pytest
import asyncio
import tracemalloc
import numpy as np
import pytalib
import gc

@pytest.mark.asyncio
async def test_high_concurrency_no_leak():
    np.random.seed(42)
    in_real = np.random.random(5000) * 100 + 10
    
    # Pre-warm function to resolve lazy loads (e.g., pandas imports inside wrappers)
    await pytalib.aio.SMA(in_real, timeperiod=14)
    await pytalib.aio.MACD(in_real, fastperiod=12, slowperiod=26, signalperiod=9)
    await pytalib.aio.BBANDS(in_real, timeperiod=5)
    
    # Setup tracemalloc to watch memory AFTER lazy loading
    tracemalloc.start()
    
    # We will simulate high concurrency by running 1000 tasks concurrently
    num_tasks = 1000
    
    # Define an async worker
    async def worker():
        # Mix of single output and tuple output indications
        await pytalib.aio.SMA(in_real, timeperiod=14)
        await pytalib.aio.MACD(in_real, fastperiod=12, slowperiod=26, signalperiod=9)
        await pytalib.aio.BBANDS(in_real, timeperiod=5)
        
    # Take a memory snapshot before
    gc.collect()
    snap_before = tracemalloc.take_snapshot()
    
    # Execute batch concurrency
    tasks = [worker() for _ in range(num_tasks)]
    await asyncio.gather(*tasks)
    
    # Take memory snapshot after
    gc.collect()
    snap_after = tracemalloc.take_snapshot()
    
    # Get top stats (differences)
    stats = snap_after.compare_to(snap_before, 'lineno')
    tracemalloc.stop()
    
    # Ignore negligible small allocations (like Python internal runtime diffs).
    # We specifically want to check if the massive NumPy arrays or capsules are leaking.
    # We sum the difference in bytes.
    total_diff_bytes = sum(stat.size_diff for stat in stats)
    
    # 5MB threshold for safety, usually a real leak of 1000 tasks doing TA-lib arrays would be hundreds of MBs
    assert total_diff_bytes < 5_000_000, f"Memory leak detected! Total difference: {total_diff_bytes / 1024 / 1024:.2f} MB"
