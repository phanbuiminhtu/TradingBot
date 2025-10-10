import os
import pandas as pd
from datetime import datetime
from vnstock import Vnstock, Quote

DATA_FOLDER = 'price_data'

def save_data_to_csv(symbol, interval, df):
    """Save DataFrame to CSV in standardized format."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    filepath = os.path.join(DATA_FOLDER, f'{symbol}_{interval}.csv')

    df = df.rename(columns={
        'time': 'datetime',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.to_csv(filepath, index=False)
    print(f"[{symbol}] Saved {len(df)} rows → {filepath}")


def fetch_from_vnstock(symbol, interval, start_date, end_date):
    """Fetch data from vnstock within custom date range."""
    try:
        stock = Quote(symbol=symbol, source="VCI")
        df = stock.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval=interval
        )

        if df is not None and not df.empty:
            save_data_to_csv(symbol, interval, df)
        else:
            print(f"[{symbol}] No data returned for {interval} between {start_date} and {end_date}")
    except Exception as e:
        print(f"Error fetching {symbol} ({interval}): {e}")


def get_close_price(symbol_list, interval, start_date, end_date):
    """Fetch close prices for symbols + VNINDEX for custom date range."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    combined_df = None

    # Step 1: Fetch and collect each symbol
    for symbol in symbol_list:
        print(f"\n=== Fetching {symbol} ({interval}) ===")
        fetch_from_vnstock(symbol, interval, start_date, end_date)

        csv_path = os.path.join(DATA_FOLDER, f"{symbol}_{interval}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["datetime"])
            df = df[["datetime", "close"]].rename(columns={"close": symbol})
            combined_df = df if combined_df is None else pd.merge(combined_df, df, on="datetime", how="outer")

    # Step 2: Fetch VNINDEX
    print("\n=== Fetching VNINDEX ===")
    fetch_from_vnstock("VNINDEX", interval, start_date, end_date)
    vn_path = os.path.join(DATA_FOLDER, f"VNINDEX_{interval}.csv")
    if os.path.exists(vn_path):
        vn_df = pd.read_csv(vn_path, parse_dates=["datetime"])
        vn_df = vn_df[["datetime", "close"]].rename(columns={"close": "VNINDEX"})
        combined_df = pd.merge(combined_df, vn_df, on="datetime", how="outer")

    # Step 3: Format and save
    if combined_df is not None and not combined_df.empty:
        combined_df = combined_df.sort_values("datetime", ascending=False)
        combined_df["datetime"] = combined_df["datetime"].dt.strftime("%m/%d/%Y")

        symbols_name = "_".join(symbol_list)
        output_filename = f"{symbols_name}_{interval}_{start_date.year}_{end_date.year}.csv"
        output_path = os.path.join(DATA_FOLDER, output_filename)

        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved combined close prices to {output_path}")
    else:
        print("⚠️ No data to save.")


# Example usage:
if __name__ == "__main__":
    # Example: get 2023 daily close prices for selected symbols
    symbols = ["GMD", "SSB", "BVH", "BSI", "DIG", "DBC", "NKG", "DGC", "PVS", "VRE"]
    start = datetime(2020, 1, 1)
    end = datetime(2025, 10, 9)

    #get_close_price(symbols, "1D", start, end)

    fetch_from_vnstock("DIG","1D",start,end)
