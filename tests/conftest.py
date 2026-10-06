import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def prices():
    """Deterministic synthetic adjusted-close panel: 300 days x 40 tickers."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    rets = rng.normal(0.0003, 0.02, size=(300, 40))
    px = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(px, index=dates, columns=[f"T{i:02d}" for i in range(40)])
