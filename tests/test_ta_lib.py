import pytest
import numpy as np
import pandas as pd
import pytalib

def test_sma_numpy():
    # 10 elements
    in_real = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    
    # Calculate SMA with period 3
    # Expected: 
    # [NaN, NaN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    out = pytalib.SMA(in_real, timeperiod=3)
    
    assert isinstance(out, np.ndarray)
    assert len(out) == 10
    
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    assert out[9] == pytest.approx(9.0)

def test_sma_pandas():
    # 10 elements
    in_real = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=list("abcdefghij"), name="prices")
    
    out = pytalib.SMA(in_real, timeperiod=5)
    
    assert isinstance(out, pd.Series)
    assert len(out) == 10
    assert out.name == "prices"
    assert out.index.tolist() == list("abcdefghij")
    
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[3])
    # SMA5 for element 4 (value 5) is sum(1,2,3,4,5)/5 = 3
    assert out.iloc[4] == pytest.approx(3.0)

def test_empty_array():
    in_real = np.array([])
    out = pytalib.SMA(in_real, timeperiod=3)
    assert len(out) == 0

def test_invalid_input():
    in_real = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(Exception):
        pytalib.SMA(in_real, timeperiod=3)
