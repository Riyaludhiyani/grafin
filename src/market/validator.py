"""Statistical market validation module for financial knowledge graph relationships.

Calculates:
- Daily percentage returns
- Date-aligned time-series joins (handling asynchronous exchange holidays)
- Appropriately lagged cross-correlations
- Empirical evidence labeling: supporting, contradicting, or inconclusive
- Enforces strict minimum sample sizes (>= 30 days)
- Emphasizes: Correlation is supporting or contradicting evidence, NEVER proof of causation.
"""

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import MIN_SAMPLE_SIZE_FOR_CORRELATION
from src.market.downloader import MarketDataDownloader

ValidationStatus = Literal["supporting", "contradicting", "inconclusive", "insufficient_data"]


class MarketValidationReport(BaseModel):
    """Structured empirical evaluation of a hypothesized graph edge against market data."""
    source_symbol: str
    target_symbol: str
    prior_sign: str
    evaluated_lag: int
    correlation_statistic: Optional[float] = None
    peak_lag: Optional[int] = None
    peak_correlation: Optional[float] = None
    sample_size: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: ValidationStatus
    explanation: str
    limitations: List[str] = Field(default_factory=list)
    causation_disclaimer: str = (
        "Empirical correlation reflects historical statistical association and is never proof of causal transmission."
    )


def compute_daily_returns(price_series: pd.Series) -> pd.Series:
    """Calculate clean daily percentage returns with dropped initial NaN."""
    if price_series is None or price_series.empty or len(price_series) < 2:
        return pd.Series(dtype=float)
    clean_series = price_series.dropna()
    returns = clean_series.pct_change().dropna()
    return returns[~returns.isin([np.inf, -np.inf])]


def align_price_returns(
    series_a: pd.Series, series_b: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """Inner join two return series on matching dates, eliminating asynchronous market holidays."""
    if series_a.empty or series_b.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Convert indices to normalized date format (YYYY-MM-DD)
    df_a = pd.DataFrame({"ret_a": series_a})
    df_a.index = pd.to_datetime(df_a.index).normalize()
    df_a = df_a[~df_a.index.duplicated(keep="last")]

    df_b = pd.DataFrame({"ret_b": series_b})
    df_b.index = pd.to_datetime(df_b.index).normalize()
    df_b = df_b[~df_b.index.duplicated(keep="last")]

    merged = df_a.join(df_b, how="inner").dropna()
    return merged["ret_a"], merged["ret_b"]


def compute_lagged_correlation(
    returns_source: pd.Series, returns_target: pd.Series, lag_days: int = 0
) -> Optional[float]:
    """Compute Pearson correlation where returns_source at time t leads returns_target at t + lag_days.

    Args:
        returns_source: Returns of the driver asset.
        returns_target: Returns of the affected asset.
        lag_days: Forward lag in trading days (0 <= lag <= 15).

    Returns:
        Pearson correlation coefficient as float, or None if sample size is insufficient.
    """
    if len(returns_source) < 5 or len(returns_target) < 5:
        return None

    if lag_days == 0:
        corr = returns_source.corr(returns_target)
    else:
        # returns_target shifted backwards by lag_days aligns date t with date t+lag_days
        target_shifted = returns_target.shift(-lag_days)
        valid_idx = target_shifted.dropna().index
        if len(valid_idx) < 5:
            return None
        corr = returns_source.loc[valid_idx].corr(target_shifted.loc[valid_idx])

    return round(float(corr), 4) if pd.notna(corr) else None


class MarketValidator:
    """Validates financial knowledge graph edges against historical market time series."""

    def __init__(self, downloader: Optional[MarketDataDownloader] = None):
        self.downloader = downloader or MarketDataDownloader()

    def validate_relationship(
        self,
        source_symbol: str,
        target_symbol: str,
        prior_sign: str = "positive",
        expected_lag_days: int = 0,
        max_lag_check: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_sample_size: int = MIN_SAMPLE_SIZE_FOR_CORRELATION,
    ) -> MarketValidationReport:
        """Validate a hypothesized graph edge against empirical market returns.

        Args:
            source_symbol: Driver ticker (e.g. 'BZ=F' for Brent).
            target_symbol: Affected ticker (e.g. 'ASIANPAINT.NS').
            prior_sign: 'positive', 'negative', or 'ambiguous'.
            expected_lag_days: Hypothesized transmission horizon in days.
            max_lag_check: Upper limit of lag search space.
            start_date: History start date string.
            end_date: History end date string.
            min_sample_size: Minimum required aligned trading days (default 30).

        Returns:
            MarketValidationReport detailing correlation, sample size, limitations, and empirical verdict.
        """
        limitations = [
            "Correlation does not establish causality; co-movements may stem from shared macro factors.",
            "Market holidays on international commodity exchanges vs Indian NSE lead to non-overlapping dates.",
        ]

        # 1. Fetch historical prices
        df_src = self.downloader.get_historical_prices(source_symbol, start_date=start_date, end_date=end_date)
        df_tgt = self.downloader.get_historical_prices(target_symbol, start_date=start_date, end_date=end_date)

        if df_src is None or df_src.empty:
            return MarketValidationReport(
                source_symbol=source_symbol,
                target_symbol=target_symbol,
                prior_sign=prior_sign,
                evaluated_lag=expected_lag_days,
                sample_size=0,
                status="insufficient_data",
                explanation=f"Historical price data unavailable or empty for source symbol '{source_symbol}'.",
                limitations=limitations,
            )

        if df_tgt is None or df_tgt.empty:
            return MarketValidationReport(
                source_symbol=source_symbol,
                target_symbol=target_symbol,
                prior_sign=prior_sign,
                evaluated_lag=expected_lag_days,
                sample_size=0,
                status="insufficient_data",
                explanation=f"Historical price data unavailable or empty for target symbol '{target_symbol}'.",
                limitations=limitations,
            )

        # 2. Compute returns and align on common trading dates
        ret_src = compute_daily_returns(df_src["Close"])
        ret_tgt = compute_daily_returns(df_tgt["Close"])
        aligned_src, aligned_tgt = align_price_returns(ret_src, ret_tgt)

        sample_size = len(aligned_src)
        if sample_size < min_sample_size:
            return MarketValidationReport(
                source_symbol=source_symbol,
                target_symbol=target_symbol,
                prior_sign=prior_sign,
                evaluated_lag=expected_lag_days,
                sample_size=sample_size,
                status="insufficient_data",
                explanation=(
                    f"Aligned sample size ({sample_size} days) is below the required minimum threshold of "
                    f"{min_sample_size} trading days to evaluate statistical relationship."
                ),
                limitations=limitations,
            )

        date_start = aligned_src.index[0].strftime("%Y-%m-%d")
        date_end = aligned_src.index[-1].strftime("%Y-%m-%d")

        # 3. Compute correlation at evaluated lag and search peak lag
        eval_corr = compute_lagged_correlation(aligned_src, aligned_tgt, lag_days=expected_lag_days)

        peak_lag = expected_lag_days
        peak_corr = eval_corr

        if eval_corr is not None:
            max_abs_corr = abs(eval_corr)
            for lag in range(0, max_lag_check + 1):
                c = compute_lagged_correlation(aligned_src, aligned_tgt, lag_days=lag)
                if c is not None and abs(c) > max_abs_corr:
                    max_abs_corr = abs(c)
                    peak_lag = lag
                    peak_corr = c

        # 4. Statistical labeling against prior sign
        corr_val = eval_corr if eval_corr is not None else 0.0
        status: ValidationStatus = "inconclusive"
        explanation = ""

        threshold = 0.15  # Meaningful economic correlation hurdle

        if prior_sign == "positive":
            if corr_val >= threshold:
                status = "supporting"
                explanation = (
                    f"Empirical correlation of +{corr_val:.2f} at lag {expected_lag_days} supports the hypothesized "
                    f"positive transmission prior over {sample_size} aligned trading days."
                )
            elif corr_val <= -threshold:
                status = "contradicting"
                explanation = (
                    f"Empirical correlation of {corr_val:.2f} at lag {expected_lag_days} contradicts the hypothesized "
                    f"positive transmission prior."
                )
            else:
                status = "inconclusive"
                explanation = (
                    f"Empirical correlation of {corr_val:.2f} is within the neutral noise band (-0.15 to +0.15); "
                    "insufficient evidence to confirm positive transmission."
                )

        elif prior_sign == "negative":
            if corr_val <= -threshold:
                status = "supporting"
                explanation = (
                    f"Empirical correlation of {corr_val:.2f} at lag {expected_lag_days} supports the hypothesized "
                    f"inverse/negative transmission prior over {sample_size} aligned trading days."
                )
            elif corr_val >= threshold:
                status = "contradicting"
                explanation = (
                    f"Empirical correlation of +{corr_val:.2f} at lag {expected_lag_days} contradicts the hypothesized "
                    f"inverse transmission prior."
                )
            else:
                status = "inconclusive"
                explanation = (
                    f"Empirical correlation of {corr_val:.2f} is within the neutral noise band (-0.15 to +0.15); "
                    "insufficient evidence to confirm inverse transmission."
                )

        else:  # ambiguous prior
            status = "inconclusive"
            explanation = (
                f"Prior is ambiguous; empirical correlation is {corr_val:.2f} with peak correlation {peak_corr} "
                f"at lag {peak_lag}."
            )

        return MarketValidationReport(
            source_symbol=source_symbol,
            target_symbol=target_symbol,
            prior_sign=prior_sign,
            evaluated_lag=expected_lag_days,
            correlation_statistic=eval_corr,
            peak_lag=peak_lag,
            peak_correlation=peak_corr,
            sample_size=sample_size,
            start_date=date_start,
            end_date=date_end,
            status=status,
            explanation=explanation,
            limitations=limitations,
        )
