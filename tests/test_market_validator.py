"""Unit tests for Phase 6: Market data downloader, return calculations, caching, and statistical validation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.market.downloader import MarketDataDownloader
from src.market.validator import (
    MarketValidationReport,
    MarketValidator,
    align_price_returns,
    compute_daily_returns,
    compute_lagged_correlation,
)


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Temporary directory for caching tests."""
    cache = tmp_path / "test_market_cache"
    cache.mkdir()
    return cache


def test_daily_returns_calculation():
    """Verify daily returns percentage computation and NaN filtering."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 110.0, 121.0, 121.0], index=dates)

    returns = compute_daily_returns(prices)

    assert len(returns) == 3
    assert pytest.approx(returns.iloc[0], rel=1e-4) == 0.10
    assert pytest.approx(returns.iloc[1], rel=1e-4) == 0.10
    assert pytest.approx(returns.iloc[2], rel=1e-4) == 0.0

    # Short/empty series returns empty series
    assert compute_daily_returns(pd.Series([100.0])).empty
    assert compute_daily_returns(pd.Series(dtype=float)).empty


def test_date_alignment_drops_mismatched_holidays():
    """Verify inner join drops exchange holiday mismatches without fabricating data."""
    dates_a = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    dates_b = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    s_a = pd.Series([0.01, 0.02, 0.03], index=dates_a)
    s_b = pd.Series([0.05, 0.06, 0.07], index=dates_b)

    aligned_a, aligned_b = align_price_returns(s_a, s_b)

    # Only Jan 2 and Jan 3 match
    assert len(aligned_a) == 2
    assert len(aligned_b) == 2
    assert list(aligned_a.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert pytest.approx(aligned_a.iloc[0]) == 0.02
    assert pytest.approx(aligned_b.iloc[0]) == 0.05


def test_lagged_correlation_math():
    """Verify lagged correlation correctly shifts lead-lag series."""
    # Construct synthetic series where Series B at t+1 mirrors Series A at t
    np.random.seed(42)
    series_a_vals = np.random.normal(0, 1, 50)
    series_b_vals = np.roll(series_a_vals, 1)  # B lags A by 1 day

    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    s_a = pd.Series(series_a_vals, index=dates)
    s_b = pd.Series(series_b_vals, index=dates)

    # Lag 1 correlation should be ~1.0
    corr_lag1 = compute_lagged_correlation(s_a, s_b, lag_days=1)
    assert corr_lag1 is not None
    assert corr_lag1 > 0.90

    # Lag 0 correlation should be significantly weaker
    corr_lag0 = compute_lagged_correlation(s_a, s_b, lag_days=0)
    assert corr_lag0 is not None
    assert corr_lag0 < corr_lag1


def test_minimum_sample_size_enforced(mock_cache_dir):
    """Verify that sample size < 30 returns 'insufficient_data' status."""
    downloader = MarketDataDownloader(cache_dir=mock_cache_dir)
    validator = MarketValidator(downloader=downloader)

    # Create dummy series of only 20 trading days
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df_small = pd.DataFrame({"Close": np.linspace(100, 120, 20)}, index=dates)

    downloader.get_historical_prices = MagicMock(return_value=df_small)

    report = validator.validate_relationship(
        source_symbol="MOCK_A",
        target_symbol="MOCK_B",
        prior_sign="positive",
        expected_lag_days=0,
        min_sample_size=30,
    )

    assert report.status == "insufficient_data"
    assert report.sample_size == 19  # 20 prices -> 19 returns
    assert "below the required minimum threshold of 30" in report.explanation


def test_supporting_evidence_labeling(mock_cache_dir):
    """Verify that empirical correlation aligned with prior yields 'supporting' label."""
    downloader = MarketDataDownloader(cache_dir=mock_cache_dir)
    validator = MarketValidator(downloader=downloader)

    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    np.random.seed(42)
    r = np.random.normal(0.001, 0.02, 60)
    df_pos_a = pd.DataFrame({"Close": 100.0 * np.cumprod(1 + r)}, index=dates)
    df_pos_b = pd.DataFrame({"Close": 200.0 * np.cumprod(1 + r)}, index=dates)

    downloader.get_historical_prices = MagicMock(side_effect=[df_pos_a, df_pos_b])

    report = validator.validate_relationship(
        source_symbol="MOCK_A",
        target_symbol="MOCK_B",
        prior_sign="positive",
        expected_lag_days=0,
    )

    assert report.status == "supporting"
    assert report.correlation_statistic is not None
    assert report.correlation_statistic > 0.15
    assert "supports the hypothesized positive transmission prior" in report.explanation
    assert "never proof of causal transmission" in report.causation_disclaimer


def test_contradicting_evidence_labeling(mock_cache_dir):
    """Verify that empirical correlation opposing prior yields 'contradicting' label."""
    downloader = MarketDataDownloader(cache_dir=mock_cache_dir)
    validator = MarketValidator(downloader=downloader)

    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    np.random.seed(42)
    r = np.random.normal(0.001, 0.02, 60)
    # Asset B moves in exact opposite percentage return direction to Asset A
    df_a = pd.DataFrame({"Close": 100.0 * np.cumprod(1 + r)}, index=dates)
    df_b = pd.DataFrame({"Close": 200.0 * np.cumprod(1 - r)}, index=dates)

    downloader.get_historical_prices = MagicMock(side_effect=[df_a, df_b])

    # Prior is positive, but empirical returns move in opposite direction
    report = validator.validate_relationship(
        source_symbol="MOCK_A",
        target_symbol="MOCK_B",
        prior_sign="positive",
        expected_lag_days=0,
    )

    assert report.status == "contradicting"
    assert report.correlation_statistic is not None
    assert report.correlation_statistic < -0.15
    assert "contradicts the hypothesized positive transmission prior" in report.explanation


def test_caching_mechanism(mock_cache_dir):
    """Verify that downloaded data is saved to cache and reloaded without new network requests."""
    downloader = MarketDataDownloader(cache_dir=mock_cache_dir)

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    fake_df = pd.DataFrame({"Close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]}, index=dates)

    # Patch yfinance Ticker history
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = fake_df
        mock_ticker.return_value = mock_instance

        # 1. First fetch: downloads and writes cache
        df_first = downloader.get_historical_prices("TEST_SYM", start_date="2024-01-01", end_date="2024-01-10", use_cache=True)
        assert df_first is not None
        assert len(df_first) == 10
        assert mock_instance.history.call_count == 1

        # 2. Second fetch: loads directly from disk cache without calling yfinance
        df_second = downloader.get_historical_prices("TEST_SYM", start_date="2024-01-01", end_date="2024-01-10", use_cache=True)
        assert df_second is not None
        assert len(df_second) == 10
        # Call count must still be 1 (zero new network calls!)
        assert mock_instance.history.call_count == 1


def test_never_fabricates_synthetic_prices(mock_cache_dir):
    """Verify that unavailable symbols return None instead of fabricating fake data."""
    downloader = MarketDataDownloader(cache_dir=mock_cache_dir)

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()  # Empty
        mock_ticker.return_value = mock_instance

        df = downloader.get_historical_prices("NONEXISTENT_TICKER")
        assert df is None

        # Empty symbol
        assert downloader.get_historical_prices("") is None
        assert downloader.get_historical_prices("   ") is None


def test_missing_close_column_handled(mock_cache_dir):
    """Verify that dataframe missing Close and Adj Close returns None without crashing."""
    downloader = MarketDataDownloader(cache_dir=mock_cache_dir)

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        # Missing 'Close'
        mock_instance.history.return_value = pd.DataFrame({"Volume": [100, 200]}, index=pd.date_range("2024-01-01", periods=2))
        mock_ticker.return_value = mock_instance

        df = downloader.get_historical_prices("BAD_COL_SYM")
        assert df is None


def test_real_market_data_smoke_test():
    """Separately labeled real integration check testing live yfinance connectivity.

    If internet is unreachable, gracefully skips without failing the offline suite.
    """
    downloader = MarketDataDownloader()
    try:
        # Download small 5-day window of USD/INR
        df = downloader.get_historical_prices("INR=X", use_cache=True, timeout=5)
        if df is None:
            pytest.skip("Yahoo Finance network returned no data (skipping live test)")
        assert not df.empty
        assert "Close" in df.columns
    except Exception as e:
        pytest.skip(f"Live network market data unreachable ({e}); skipping live test.")
