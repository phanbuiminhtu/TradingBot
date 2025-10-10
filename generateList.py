# filename: generate_icb_stock_lists.py

from vnstock import Listing

def generate_icb_stock_lists_file(output_file='icb_stock_lists.py'):
    listing = Listing()
    df = listing.symbols_by_industries()

    df = df[['symbol', 'icb_code4']].dropna()
    df = df[df['symbol'].apply(lambda x: isinstance(x, str) and x.isalnum())]

    grouped = df.groupby('icb_code4')['symbol'].apply(list)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Auto-generated ICB stock list\n\n")
        for icb_code, symbols in grouped.items():
            list_name = f"ICB_{str(icb_code)}"
            symbol_list_str = repr(sorted(symbols))
            f.write(f"{list_name} = {symbol_list_str}\n")

    print(f"ICB stock list saved to {output_file}")

if __name__ == '__main__':
    generate_icb_stock_lists_file()
    #listing = Listing()
    #name = listing.symbols_by_industries()
    #unique_icb = name[['icb_name4', 'icb_code4']].dropna().drop_duplicates()

   #print(unique_icb.sort_values(by='icb_code4').to_string(index=False))
