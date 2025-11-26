import pandas as pd
from vnstock import Finance

# --- Configuration ---
symbol = "PVD" 
source_name = 'VCI'
period_type = 'year'
language = 'en' 

# Define the common file saving parameters
export_params = {
    'index': False,
    # Use standard comma delimiter for CSV
    'sep': ',',      
    # Use 'utf-8-sig' (UTF-8 with BOM) for best Excel compatibility 
    'encoding': 'utf-8-sig' 
}

print(f"--- Fetching {period_type.capitalize()} financial data for {symbol} ---")

try:
    # --- 1. Data Retrieval ---
    print("Fetching Income Statement...")
    income_df = Finance(symbol=symbol, source=source_name).income_statement(period=period_type, lang=language)
    
    print("Fetching Balance Sheet...")
    balance_df = Finance(symbol=symbol, source=source_name).balance_sheet(period=period_type, lang=language)
    
    print("Fetching Cash Flow Statement...")
    cash_flow_df = Finance(symbol=symbol, source=source_name).cash_flow(period=period_type, lang=language)
    
    # --- 2. Export to CSV/TSV ---
    
    data_frames = {
        'income_statement': income_df,
        'balance_sheet': balance_df,
        'cash_flow_statement': cash_flow_df
    }

    print("\n--- Exporting Files ---")
    
    all_empty = True
    for name, df in data_frames.items():
        if df.empty:
            print(f"⚠️ Warning: {name.replace('_', ' ').title()} DataFrame is empty. Skipping export.")
        else:
            # File name with standard .csv extension
            file_name = f'{symbol}_{name}_{period_type}.csv'
            
            # Save the data using 'utf-8-sig' and comma delimiter
            df.to_csv(file_name, **export_params)
            
            print(f"✅ Saved {name.replace('_', ' ')} data to **{file_name}**")
            all_empty = False

    if all_empty:
         print(f"\n❌ Final Status: No financial data could be retrieved for symbol {symbol}.")
    else:
        print("\n🎉 Final Status: All files saved with **'utf-8-sig'** encoding and **comma** delimiter.")
        print("💡 You should now be able to **double-click** the CSV files and have Excel correctly display Vietnamese characters and columns.")

except Exception as e:
    print(f"\n❌ A critical error occurred during data retrieval or export: {e}")