import pytest
import pytafast


def test_talib_error_inheritance():
    # Verify TALibError is exposed and inherits from Exception
    assert hasattr(pytafast, "TALibError")
    assert issubclass(pytafast.TALibError, Exception)


def test_talib_error_raising_and_catching():
    # Verify we can raise and catch TALibError
    with pytest.raises(pytafast.TALibError) as excinfo:
        raise pytafast.TALibError("A TA-Lib internal error occurred")
    assert "A TA-Lib internal error occurred" in str(excinfo.value)
