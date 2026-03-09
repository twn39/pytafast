import asyncio
import tracemalloc
import numpy as np
import pytafast
import gc
import os
import psutil
import pytest

def get_process_memory():
    """Returns the current process RSS memory in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

async def run_workload(n_iterations, data):
    """Executes a mix of technical indicators multiple times."""
    for _ in range(n_iterations):
        # We use various return types (single array, tuples)
        await pytafast.aio.SMA(data, timeperiod=14)
        await pytafast.aio.MACD(data, fastperiod=12, slowperiod=26, signalperiod=9)
        await pytafast.aio.BBANDS(data, timeperiod=5)
        await pytafast.aio.STOCH(data, data, data)

def test_async_memory_leak_final():
    """
    Final robust memory leak test that:
    1. Ignores initial memory pool spikes.
    2. Analyzes steady-state RSS growth (System/C++ level).
    3. Analyzes Python-level allocation diffs (Python objects).
    """
    asyncio.run(_test_async_memory_leak_impl())

async def _test_async_memory_leak_impl():
    # Setup data
    np.random.seed(42)
    data = np.random.random(20000).astype(np.float64)
    
    # 1. Warm up
    await run_workload(50, data)
    gc.collect()
    
    # 2. Baseline for Python-level tracking
    tracemalloc.start()
    snap_start = tracemalloc.take_snapshot()
    
    # 3. Iterative Sampling
    samples = []
    num_checkpoints = 10
    batch_size = 100
    
    print(f"\n[Memory Steady-State Analysis]")
    
    for c in range(num_checkpoints):
        # Run a batch of iterations
        tasks = [run_workload(1, data) for _ in range(batch_size)]
        await asyncio.gather(*tasks)
        
        gc.collect()
        mem = get_process_memory()
        samples.append(mem)
        print(f"Checkpoint {c+1}: RSS = {mem / 1024 / 1024:.2f} MB")
        
    # 4. Python-level post-measurement
    snap_end = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    # 5. Analysis
    # We ignore the first 5 samples as warming/pool expansion for RSS
    steady_samples = samples[5:]
    steady_diffs = [steady_samples[i] - steady_samples[i-1] for i in range(1, len(steady_samples))]
    avg_steady_growth = np.mean(steady_diffs)
    
    # Python diff analysis
    stats = snap_end.compare_to(snap_start, "lineno")
    py_diff = sum(stat.size_diff for stat in stats)
    
    print(f"Average RSS growth in steady state: {avg_steady_growth / 1024 / 1024:.3f} MB per {batch_size} iterations")
    print(f"Python-level allocation diff: {py_diff / 1024 / 1024:.3f} MB")
    
    # Thresholds
    # RSS: A real leak of 1000 items would be ~400MB. 5MB per batch is very safe.
    assert avg_steady_growth < 5 * 1024 * 1024, f"Linear RSS leak detected! Avg growth: {avg_steady_growth / 1024 / 1024:.2f} MB per batch"
    
    # Python: Python level growth should be minimal
    assert py_diff < 2 * 1024 * 1024, f"Python objects leaked: {py_diff / 1024 / 1024:.2f} MB"
