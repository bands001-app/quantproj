"""Optional: pull real prices into the CSV contract ``data.load_csv`` expects.

Kept out of the package dependencies deliberately -- the library and its tests
must run offline and deterministically. Install ``yfinance`` yourself if you
want this.

    pip install yfinance
    python scripts/fetch_data.py --tickers SPY TLT GLD DBC EFA --out data/prices.csv
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data/prices.csv")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install yfinance to use this script") from exc

    raw = yf.download(
        args.tickers, start=args.start, end=args.end, auto_adjust=True, progress=False
    )
    close = raw["Close"] if "Close" in getattr(raw, "columns", []) else raw  # noqa: SIM401
    close = close.dropna(how="all").ffill().dropna()
    close.index.name = "date"
    close.to_csv(args.out)
    print(f"wrote {close.shape[0]} rows x {close.shape[1]} cols -> {args.out}")


if __name__ == "__main__":
    main()
