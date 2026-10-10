"""Market data downloader with persistent local caching, timeouts, and symbol verification.

Ensures:
- Local disk caching in data/market_cache/
- Configurable date ranges and symbols
- Verifies returned columns ('Close' or 'Adj Close')
- Never fabricates prices or invents synthetic market values
- Graceful offline and network failure handling
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import yfinance as yf

from src.config import MARKET_CACHE_DIR

logger = logging.getLogger(__name__)


def sanitize_filename(symbol: str) -> str:
    """Sanitize market ticker symbol for safe file storage."""
    return re.sub(r"[^\w\-.]", "_", symbol)


class MarketDataDownloader:
    """Downloads daily market time series via yfinance with transparent local caching."""

    def __init__(self, cache_dir: Union[str, Path] = MARKET_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_paths(self, symbol: str, start_date: str, end_date: str) -> Tuple[Path, Path]:
        """Generate cache data CSV path and metadata JSON path."""
        clean_sym = sanitize_filename(symbol)
        date_hash = hashlib.md5(f"{start_date}_{end_date}".encode()).hexdigest()[:8]
        csv_path = self.cache_dir / f"{clean_sym}_{start_date}_{end_date}_{date_hash}.csv"
        meta_path = self.cache_dir / f"{clean_sym}_{start_date}_{end_date}_{date_hash}.meta.json"
        return csv_path, meta_path

    def get_historical_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        timeout: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical daily price time series for a given ticker symbol.

        Args:
            symbol: Yahoo Finance ticker (e.g. 'BZ=F', 'INR=X', '^NSEI', 'ASIANPAINT.NS').
            start_date: Start date string 'YYYY-MM-DD' (defaults to 1 year ago).
            end_date: End date string 'YYYY-MM-DD' (defaults to today).
            use_cache: If True, reads from disk cache if already downloaded.
            timeout: Network request timeout in seconds.

        Returns:
            DataFrame with DatetimeIndex and ['Open', 'High', 'Low', 'Close', 'Volume'],
            or None if the symbol is invalid, unavailable, or empty.
        """
        if not symbol or not symbol.strip():
            logger.warning("Empty ticker symbol supplied to get_historical_prices.")
            return None

        clean_symbol = symbol.strip()
        end_dt = datetime.now() if not end_date else datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = (end_dt - timedelta(days=365)) if not start_date else datetime.strptime(start_date, "%Y-%m-%d")

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        csv_path, meta_path = self._get_cache_paths(clean_symbol, start_str, end_str)

        # 1. Check local cache
        if use_cache and csv_path.exists() and meta_path.exists():
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if not df.empty and ("Close" in df.columns or "Adj Close" in df.columns):
                    logger.info(f"Loaded {clean_symbol} from disk cache ({len(df)} rows).")
                    return df
            except Exception as e:
                logger.warning(f"Error reading cache file {csv_path}: {e}. Re-downloading.")

        # 2. Download via yfinance
        try:
            logger.info(f"Downloading market data for {clean_symbol} from {start_str} to {end_str}...")
            ticker_obj = yf.Ticker(clean_symbol)
            df = ticker_obj.history(start=start_str, end=end_str, timeout=timeout)

            if df is None or df.empty:
                logger.warning(f"No price data returned by Yahoo Finance for symbol: '{clean_symbol}'")
                return None

            # Standardize index and ensure Close column
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if "Close" not in df.columns:
                if "Adj Close" in df.columns:
                    df["Close"] = df["Adj Close"]
                else:
                    logger.warning(f"Neither 'Close' nor 'Adj Close' column found for {clean_symbol}.")
                    return None

            # Drop duplicate dates and sort chronologically
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            # Save to local cache
            df.to_csv(csv_path)
            metadata = {
                "symbol": clean_symbol,
                "start_date": start_str,
                "end_date": end_str,
                "downloaded_at": datetime.utcnow().isoformat() + "Z",
                "source": "Yahoo Finance (yfinance)",
                "row_count": len(df),
                "columns": list(df.columns),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            return df

        except Exception as e:
            logger.warning(f"Failed to download market data for symbol '{clean_symbol}': {e}")
            return None

    def get_latest_close(self, symbol: str) -> Optional[Tuple[float, str]]:
        """Get the latest available close price and its timestamp.

        Returns:
            Tuple of (close_price, date_string) or None if unavailable.
        """
        df = self.get_historical_prices(symbol)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        last_dt = df.index[-1].strftime("%Y-%m-%d")
        last_price = float(df["Close"].iloc[-1])
        return last_price, last_dt
