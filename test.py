from vnstock import Finance
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

'''
finance = Finance(symbol='SHS', source='VCI')
ratio = finance.ratio(period='quarter', lang='vi')


def calculate_peg_from_ratio_df(ratio_df, symbol):
    # column names trong DataFrame trả về từ finance.ratio()
    col_cp   = ('Meta', 'CP')
    col_year = ('Meta', 'Năm')
    col_k    = ('Meta', 'Kỳ')
    col_pe   = ('Chỉ tiêu định giá', 'P/E')
    col_eps  = ('Chỉ tiêu định giá', 'EPS (VND)')

    # kiểm tra tồn tại các cột bắt buộc
    for c in (col_cp, col_year, col_k, col_pe, col_eps):
        if c not in ratio_df.columns:
            raise KeyError(f"Không tìm thấy cột {c} trong ratio_df.columns")

    # lọc ra các hàng cho symbol
    df_sym = ratio_df[ ratio_df[col_cp] == symbol ].copy()
    if df_sym.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho symbol={symbol}")

    # ép kiểu số an toàn
    df_sym[col_year] = pd.to_numeric(df_sym[col_year], errors='coerce')
    df_sym[col_k]    = pd.to_numeric(df_sym[col_k], errors='coerce')
    df_sym[col_pe]   = pd.to_numeric(df_sym[col_pe], errors='coerce')
    df_sym[col_eps]  = pd.to_numeric(df_sym[col_eps], errors='coerce')

    # tạo key để sắp xếp (ví dụ 2025*10 + 2 => 20252)
    df_sym['_sort_key'] = (df_sym[col_year].fillna(0).astype(int) * 10 +
                           df_sym[col_k].fillna(0).astype(int))
    df_sym = df_sym.sort_values('_sort_key', ascending=False)

    # lấy 2 kỳ gần nhất
    cur = df_sym.iloc[0]
    prev = df_sym.iloc[1] if len(df_sym) > 1 else None

    pe = cur[col_pe]
    eps_cur = cur[col_eps]
    eps_prev = prev[col_eps] if prev is not None else np.nan

    # tính tăng trưởng EPS (%) và PEG
    eps_growth_pct = np.nan
    peg = np.nan
    if pd.notnull(eps_prev) and eps_prev != 0:
        eps_growth_pct = (eps_cur - eps_prev) / eps_prev * 100
        if pd.notnull(pe) and eps_growth_pct != 0:
            peg = pe / eps_growth_pct

    return {
        "symbol": symbol,
        "period_current": (int(cur[col_year]), int(cur[col_k])),
        "P/E": float(pe) if pd.notnull(pe) else None,
        "EPS_current": float(eps_cur) if pd.notnull(eps_cur) else None,
        "EPS_prev": float(eps_prev) if pd.notnull(eps_prev) else None,
        "EPS_growth_pct": float(eps_growth_pct) if pd.notnull(eps_growth_pct) else None,
        "PEG": float(peg) if pd.notnull(peg) else None
    }

        
print(calculate_peg_from_ratio_df(ratio, 'SHS'))
'''

end = datetime(2025, 10, 10)
start = end - relativedelta(years=5)
print(start)
