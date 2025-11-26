import textwrap
import pandas as pd
import google.generativeai as genai
from vnstock import Finance
from IPython.display import display, Markdown
from getData import fetch_from_vnstock
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from TechnicalIndicator import detect_big_money
import numpy as np
# Thư viện mới để vẽ biểu đồ và xử lý ảnh
import matplotlib.pyplot as plt
import io
from PIL import Image

# --- 1. SETUP AND CONFIGURATION ---

def configure_api():
    """
    Configures the Google Generative AI API.
    """
    try:
        # Thay thế "YOUR_API_KEY" bằng khóa API thực của bạn
        api_key = "YOUR_API_KEY"
        if not api_key or api_key == "YOUR_API_KEY":
            print("ERROR: GOOGLE_API_KEY is not set or is a placeholder.")
            print("Please set your API key to proceed.")
            return None
        genai.configure(api_key=api_key)
        # Sử dụng model Pro để có khả năng phân tích hình ảnh tốt hơn
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"An error occurred during API configuration: {e}")
        return None

def to_markdown(text):
  """Formats text for nice display in notebooks."""
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# --- 2. AGENT DEFINITIONS (AS FUNCTIONS) ---

def agent_gather_data(symbol: str) -> dict:
    """
    Agent 1: Data Gatherer (Fundamental).
    """
    print(f"📈 [Data Agent] Đang lấy dữ liệu tài chính cho {symbol}...")
    try:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        income_df = Finance(symbol=symbol, source='VCI').income_statement(period='quarter', lang='vi')
        balance_df = Finance(symbol=symbol, source='VCI').balance_sheet(period='quarter', lang='vi')
        cash_flow_df = Finance(symbol=symbol, source='VCI').cash_flow(period='quarter', lang='vi')
        ratio_df = Finance(symbol=symbol, source='VCI').ratio(period='quarter', lang='vi')

        print(f"✅ [Data Agent] Đã lấy dữ liệu cơ bản thành công cho {symbol}.")
        return {
            "income_statement": income_df,
            "balance_sheet": balance_df,
            "cash_flow": cash_flow_df,
            "ratios": ratio_df
        }
    except Exception as e:
        error_message = f"❌ [Data Agent] Lỗi khi lấy dữ liệu cơ bản cho {symbol}: {e}"
        print(error_message)
        return {}

# =================================================================
# >>> AGENT ĐÃ ĐƯỢC NÂNG CẤP HOÀN TOÀN <<<
# =================================================================
def agent_technical_analysis(model, symbol: str) -> str:
    """
    Agent 1.5: Technical Analyst (Image-based + MCDX).
    Reads price data from CSV, plots Price, Volume, and MCDX,
    and sends chart to Gemini AI for detailed analysis.
    """
    print(f"📉 [Technical Agent] Đang tạo và phân tích biểu đồ kỹ thuật cho {symbol}...")

    end = datetime.now()
    start = end - relativedelta(years=5)
    fetch_from_vnstock(symbol,"1D", start, end)
    file_name = f"price_data/{symbol}_1D.csv"

    try:
        # 1️⃣ Read & prepare data
        price_df = pd.read_csv(file_name)
        price_df['datetime'] = pd.to_datetime(price_df['datetime'])
        price_df = price_df.sort_values('datetime', ascending=True)

        # 2️⃣ Add MCDX smart money data
        price_df = detect_big_money(price_df)

        # 3️⃣ Create the chart
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True,
            gridspec_kw={'height_ratios': [3, 1, 1]}
        )
        fig.suptitle(f'{symbol} — Giá, Khối lượng & MCDX (5 năm)', fontsize=16)

        # === PRICE CHART ===
        ax1.plot(price_df['datetime'], price_df['close'], label='Giá đóng cửa', color='blue')
        ax1.set_ylabel('Giá (VND)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # === VOLUME CHART ===
        ax2.bar(price_df['datetime'], price_df['volume'], label='Khối lượng', color='gray', alpha=0.6)
        ax2.set_ylabel('Khối lượng')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # === MCDX CHART ===
        ax3.set_facecolor("#f9f9f9")

        # Green base background
        ax3.bar(price_df['datetime'], 20, color='green', width=0.8, alpha=0.15)

        # Retailers (green)
        ax3.bar(price_df['datetime'], price_df['RSI_Retailer'], color='green', width=0.8, alpha=0.6, label='Retailers')

        # Hot Money (yellow)
        ax3.bar(price_df['datetime'], price_df['RSI_HotMoney'], color='yellow', width=0.8, alpha=0.6, label='Hot Money')

        # Bankers (red/fuchsia depending on MA)
        colors = np.where(price_df['RSI_Banker'] >= price_df['Banker_MA'], 'red', 'fuchsia')
        ax3.bar(price_df['datetime'], price_df['RSI_Banker'], color=colors, width=0.8, alpha=0.8, label='Bankers')

        # Banker MA line (black)
        ax3.plot(price_df['datetime'], price_df['Banker_MA'], color='black', linewidth=1.2, label='Banker MA')

        # Dashed levels (5, 10, 15, 20)
        for level in [5, 10, 15, 20]:
            ax3.axhline(y=level, color="#AD34CB", linestyle="--", linewidth=1, alpha=0.8)

        ax3.set_ylim(0, 22)
        ax3.set_ylabel('MCDX')
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 4️⃣ Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        img = Image.open(buf)
        plt.show()

        # 5️⃣ Gemini AI analysis prompt
        prompt = f"""
        Bạn là một Chuyên viên Phân tích Kỹ thuật cao cấp. Dựa vào hình ảnh biểu đồ giá, khối lượng, và MCDX trong 5 năm của cổ phiếu {symbol}, hãy đưa ra một phân tích chi tiết:

        1. **Xu hướng dài hạn (Multi-year Trend):** Xác định xu hướng chính trong toàn bộ giai đoạn (tăng, giảm, đi ngang).
        2. **Các chu kỳ chính:** Cổ phiếu đã trải qua những chu kỳ tăng/giảm giá lớn nào?
        3. **Vùng hỗ trợ/kháng cự dài hạn:** Xác định các vùng giá quan trọng trong quá khứ.
        4. **Phân tích khối lượng và MCDX:** 
           - Giai đoạn nào có sự tích lũy của dòng tiền lớn (Bankers)?
           - Khi Hot Money tăng, giá phản ứng ra sao?
           - Có giai đoạn phân phối mạnh nào (Bankers rút vốn) không?
        5. **Kết luận tổng quan:** Vị thế hiện tại của cổ phiếu trong chu kỳ là gì (đầu, giữa, hay cuối)? Tiềm năng hoặc rủi ro dài hạn?
        """

        print(f"✅ [Technical Agent] Đã tạo biểu đồ, đang gửi cho AI phân tích...")
        response = model.generate_content([prompt, img])

        buf.close()
        print(f"✅ [Technical Agent] Đã hoàn thành phân tích kỹ thuật cho {symbol}.")
        return response.text

    except FileNotFoundError:
        error_message = f"⚠️ [Technical Agent] Không tìm thấy file dữ liệu giá: '{file_name}'."
        print(error_message)
        return error_message

    except Exception as e:
        error_message = f"❌ [Technical Agent] Lỗi khi phân tích kỹ thuật cho {symbol}: {e}"
        print(error_message)
        return error_message


def agent_analyze_financials(model, financial_data_dict: dict, technical_analysis_report: str, symbol: str) -> str:
    """
    Agent 2: Financial Analyst (Synthesizer).
    """
    print(f"📑 [Analyst Agent] Bắt đầu phân tích tuần tự cho {symbol}...")
    all_analyses = []

    # --- Phân tích các báo cáo tài chính (đã rút gọn cho dễ đọc) ---
    print("   [1/4] Phân tích Báo cáo kết quả kinh doanh...")
    income_df_str = financial_data_dict['income_statement'].to_string()
    prompt1 = f"""Bạn là chuyên gia phân tích tài chính trong ngành sản xuất nhựa xây dựng và hạ tầng (ống nhựa, phụ kiện nhựa). 
Dựa trên bảng dữ liệu có các cột:

{income_df_str}

Hãy thực hiện các bước phân tích sau:

1. **Tăng trưởng kinh doanh:**
   - Phân tích xu hướng tăng trưởng doanh thu và lợi nhuận trong 3–5 năm.
   - So sánh với các doanh nghiệp cùng ngành (BMP, DNP, Đại Đồng Tiến).
   - Xác định chu kỳ tăng trưởng liên quan đến đầu tư công hoặc xây dựng dân dụng.

2. **Biên lợi nhuận và cấu trúc chi phí:**
   - Tính toán và nhận xét Biên lợi nhuận gộp = Lãi gộp / Doanh thu.
   - Đánh giá biến động biên gộp do giá hạt nhựa (PVC, HDPE, PP-R).
   - Phân tích tỷ trọng chi phí bán hàng và quản lý/doanh thu (%).
   - Xem xét chi phí tài chính và ảnh hưởng của lãi vay.

3. **Hiệu quả sinh lời:**
   - Tính Biên lợi nhuận ròng = Lợi nhuận sau thuế / Doanh thu.
   - Phân tích ROE và ROA nếu có dữ liệu bổ sung.
   - Đánh giá khả năng chuyển hóa doanh thu thành lợi nhuận.

4. **Kết luận:**
   - Nhận định năng lực cạnh tranh, sức khỏe lợi nhuận, và rủi ro biên lợi nhuận.
   - Chỉ ra điểm mạnh (quy mô, thương hiệu, kênh phân phối) và điểm yếu (chi phí nguyên liệu, cạnh tranh giá).
"""

    all_analyses.append(f"### 1. Phân tích Kết quả Kinh doanh\n{model.generate_content(prompt1).text}")

    print("   [2/4] Phân tích Bảng cân đối kế toán...")
    balance_df_str = financial_data_dict['balance_sheet'].to_string()
    prompt2 = f"""Bạn là chuyên gia tài chính doanh nghiệp trong ngành sản xuất nhựa xây dựng và hạ tầng. 
Dựa trên bảng dữ liệu có các cột:

{balance_df_str}

Hãy thực hiện phân tích chi tiết:

1. **Cấu trúc tài sản:**
   - Phân tích tỷ trọng tài sản ngắn hạn và dài hạn.
   - Đặc biệt đánh giá hàng tồn kho và khoản phải thu — hai chỉ tiêu ảnh hưởng mạnh đến dòng tiền.
   - So sánh vòng quay hàng tồn kho và khoản phải thu với chuẩn ngành (nếu có dữ liệu nhiều năm).

2. **Tài sản cố định & đầu tư:**
   - Phân tích quy mô đầu tư máy móc, nhà xưởng, và năng lực sản xuất.
   - Đánh giá mức tăng tài sản cố định so với tăng trưởng doanh thu → đo hiệu quả đầu tư.

3. **Nguồn vốn và đòn bẩy:**
   - Tính tỷ lệ Nợ/Tổng tài sản và Nợ/Vốn chủ sở hữu (Debt Ratio & D/E).
   - Đánh giá khả năng tự chủ tài chính, mức đòn bẩy và rủi ro thanh khoản.
   - Theo dõi biến động nợ vay trong các giai đoạn mở rộng sản xuất.

4. **Kết luận:**
   - Đưa ra nhận định về sức khỏe tài chính, khả năng quản trị vốn lưu động, và an toàn tài chính của doanh nghiệp.
   - Chỉ ra các rủi ro tiềm ẩn (nếu hàng tồn kho hoặc phải thu tăng nhanh hơn doanh thu).
"""

    all_analyses.append(f"### 2. Phân tích Bảng cân đối kế toán\n{model.generate_content(prompt2).text}")

    print("   [3/4] Phân tích Báo cáo lưu chuyển tiền tệ...")
    cash_flow_df_str = financial_data_dict['cash_flow'].to_string()
    prompt3 = f"""Bạn là chuyên gia phân tích dòng tiền trong ngành sản xuất nhựa công nghiệp và xây dựng. 
Dựa trên bảng dữ liệu có các cột:

{cash_flow_df_str}

Hãy thực hiện phân tích sau:

1. **Dòng tiền kinh doanh (CFO):**
   - So sánh CFO với lợi nhuận sau thuế để đánh giá chất lượng lợi nhuận.
   - Nếu CFO thấp hơn lợi nhuận → xem xét nguyên nhân (tồn kho tăng, khoản phải thu tăng).
   - Đánh giá xu hướng CFO qua các năm để xác định tính ổn định.

2. **Dòng tiền đầu tư (CFI):**
   - Phân tích mức đầu tư vào tài sản cố định và mở rộng nhà máy.
   - Đánh giá liệu doanh nghiệp có tự tài trợ được đầu tư bằng dòng tiền kinh doanh hay phải vay thêm.

3. **Dòng tiền tài chính (CFF):**
   - Theo dõi mức độ vay nợ mới, trả nợ và chi trả cổ tức.
   - Nhận định chính sách tài chính: thận trọng, ổn định hay mở rộng bằng vay nợ.

4. **Tổng hợp và kết luận:**
   - Đánh giá khả năng tạo tiền thực tế, mức độ bền vững tài chính, và rủi ro dòng tiền.
   - Đưa ra kết luận ngắn gọn: doanh nghiệp có đang vận hành cân bằng giữa tăng trưởng – đầu tư – tài trợ hay không.
"""

    all_analyses.append(f"### 3. Phân tích Lưu chuyển tiền tệ\n{model.generate_content(prompt3).text}")

    # --- Final Step: Synthesis and Conclusion ---
    print("   [4/4] Tổng hợp Phân tích Cơ bản và Kỹ thuật...")
    try:
        ratios_df_str = financial_data_dict['ratios'].to_string()
        previous_analyses = "\n\n".join(all_analyses)
        prompt4 = f"""
        Bạn là Chuyên viên Phân tích Đầu tư cao cấp, kết hợp cả phân tích cơ bản và kỹ thuật.
        Nhiệm vụ của bạn là đưa ra một kết luận cuối cùng cho năm 2026 (4 tháng đầu năm) về cổ phiếu {symbol}.

        **Phần 1: Các phân tích chi tiết về tài chính doanh nghiệp (Phân tích cơ bản):**
        {previous_analyses}

        **Phần 2: Phân tích biểu đồ dài hạn (Phân tích kỹ thuật):**
        {technical_analysis_report}

        **Phần 3: Dữ liệu về các chỉ số tài chính quan trọng:**
        {ratios_df_str}

        ---
        **HƯỚNG DẪN TỔNG HỢP CUỐI CÙNG:**
        Dựa trên TẤT CẢ thông tin trên, hãy viết một báo cáo tổng hợp có cấu trúc:

        **1. Luận điểm đầu tư tổng hợp:**
           - Kết hợp cả hai góc nhìn, tóm tắt câu chuyện đầu tư chính của {symbol} hiện tại là gì?

        **2. Đánh giá theo góc nhìn cơ bản (Fundamental):**
           - **Cơ hội:** Điểm sáng nhất về tài chính là gì?
           - **Rủi ro:** Rủi ro lớn nhất về tài chính là gì?

        **3. Đánh giá theo góc nhìn kỹ thuật (Technical):**
            - **Tín hiệu Tích cực:** Xu hướng dài hạn, vùng hỗ trợ mạnh,...
            - **Tín hiệu Tiêu cực:** Vùng kháng cự mạnh, mẫu hình giá xấu,...

        **4. Kết luận và Khuyến nghị (4 tháng):**
           - **Kết hợp tất cả các yếu tố**, đưa ra đánh giá cuối cùng: Cổ phiếu này đang **HẤP DẪN**, **TRUNG LẬP**, hay **KHÔNG HẤP DẪN**.
           - **Giải thích rõ ràng** lý do cho khuyến nghị của bạn.
        """
        final_response = model.generate_content(prompt4)
        print(f"✅ [Analyst Agent] Đã hoàn thành phân tích tổng hợp cho {symbol}.")
        return final_response.text
    except Exception as e:
        error_message = f"❌ [Analyst Agent] Lỗi trong quá trình tổng hợp phân tích: {e}"
        print(error_message)
        return error_message

def agent_generate_investment_summary(model, analysis_report: str, symbol: str) -> str:
    # (Hàm này không thay đổi)
    print(f"✍️ [Advisor Agent] Đang tạo tóm tắt đầu tư cho {symbol}...")
    prompt = f"""
    Bạn là một Cố vấn Đầu tư. Hãy tổng hợp báo cáo phân tích chi tiết sau đây cho công ty {symbol} thành một bản tóm tắt đầu tư rõ ràng, có thể hành động.
    Báo cáo Phân tích Chi tiết: {analysis_report}
    ---
    **NHIỆM VỤ:**
    Viết một bản tóm tắt đầu tư cuối cùng theo cấu trúc:
    **1. Tóm tắt (2-3 câu):** Mô tả ngắn gọn sức khỏe tài chính và tình hình giá cổ phiếu.
    **2. Điểm mạnh chính (Luận điểm Tăng giá):** Liệt kê 2-3 điểm tích cực nhất.
    **3. Điểm yếu/Rủi ro chính (Luận điểm Giảm giá):** Liệt kê 2-3 rủi ro lớn nhất.
    **4. Kết luận & Khuyến nghị:** Đưa ra luận điểm đầu tư rõ ràng và lý do.
    """
    try:
        response = model.generate_content(prompt)
        print(f"✅ [Advisor Agent] Đã hoàn thành tóm tắt cho {symbol}.")
        return response.text
    except Exception as e:
        return f"❌ [Advisor Agent] Lỗi trong quá trình tạo tóm tắt: {e}"


# --- 3. MAIN ORCHESTRATION ---

def main(symbol: str):
    stock_symbol_to_analyze = symbol
    lines = []

    # --- 1. HEADER ---
    lines.append("--- Bắt đầu Phân tích Cổ phiếu Toàn diện ---")
    lines.append("")

    model = configure_api()
    if model is None:
        return

    financial_data_dictionary = agent_gather_data(stock_symbol_to_analyze)
    if not financial_data_dictionary:
        return

    technical_report = agent_technical_analysis(model, stock_symbol_to_analyze)
    if "Lỗi" in technical_report and "Không tìm thấy file" not in technical_report:
        return

    detailed_analysis = agent_analyze_financials(model, financial_data_dictionary, technical_report, stock_symbol_to_analyze)
    if "Lỗi" in detailed_analysis:
        return

    investment_summary = agent_generate_investment_summary(model, detailed_analysis, stock_symbol_to_analyze)
    if "Lỗi" in investment_summary:
        return

    # --- 2. ADD SECTIONS ---
    lines.append("==================================================")
    lines.append("          BÁO CÁO PHÂN TÍCH TOÀN DIỆN")
    lines.append("==================================================")
    lines.append("")  # thêm dòng trống
    lines.append(str(detailed_analysis).replace("\\n", "\n"))  # ép xuống dòng nếu có chuỗi \n

    lines.append("")
    lines.append("==================================================")
    lines.append("         TÓM TẮT TỪ CỐ VẤN ĐẦU TƯ")
    lines.append("==================================================")
    lines.append("")  # dòng trống
    lines.append(str(investment_summary).replace("\\n", "\n"))

    # --- 3. TẠO FOLDER RESULT ---
    os.makedirs("result", exist_ok=True)

    # --- 4. THÊM NGÀY HIỆN TẠI VÀO TÊN FILE ---
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join("result", f"{stock_symbol_to_analyze}_report_{today}.txt")

    # --- 5. GHI FILE VỚI XUỐNG DÒNG RÕ RÀNG ---
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✅ Báo cáo đã được lưu tại: {output_path}")

if __name__ == '__main__':
    # <<< THAY ĐỔI MÃ CỔ PHIẾU BẠN MUỐN PHÂN TÍCH TẠI ĐÂY >>>
    main("HII")
