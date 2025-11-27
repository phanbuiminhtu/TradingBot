import google.generativeai as genai
from vnstock import Company
from serpapi import Client
import datetime
import os
from dotenv import load_dotenv
import re
from datetime import date

# ====== CẤU HÌNH API ======
load_dotenv()
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
SERP_API_KEY = os.environ.get("SERP_API_KEY")   # <<--- THAY API KEY CỦA BẠN

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Khởi tạo SerpAPI Client
serp = Client(api_key=SERP_API_KEY)

# ====== AI AGENT ======
def generate_search_queries(stock_ticker, company_name, industry, start_date):
    prompt = f"""
Dựa trên mã cổ phiếu "{stock_ticker}", tên công ty "{company_name}", và ngành "{industry}", hãy tạo danh sách **8–10 từ khóa tìm kiếm tin tức định tính và vĩ mô** phục vụ ra quyết định đầu tư.

**YÊU CẦU ĐẶC BIỆT:**
* **TUYỆT ĐỐI KHÔNG** tìm kiếm về: Báo cáo tài chính, doanh thu, lợi nhuận, cổ tức, EPS, P/E, tăng/giảm giá cổ phiếu, phân tích kỹ thuật (vì tôi đã có dữ liệu này).
* **TẬP TRUNG HOÀN TOÀN** vào các nhóm chủ đề sau:

1.  **Chính sách & Vĩ mô Ngành:**
    * Các thông tư, nghị định, quy hoạch mới của Chính phủ liên quan đến ngành "{industry}".
    * Thuế (thuế chống bán phá giá, thuế xuất nhập khẩu, VAT), lãi suất ưu đãi, hoặc rào cản thương mại.
2.  **Sự kiện Doanh nghiệp (Phi Tài chính):**
    * Dự án mới, khởi công, trúng thầu, khánh thành nhà máy/dây chuyền.
    * M&A (thâu tóm, sáp nhập), thoái vốn, hợp tác chiến lược, ký kết hợp đồng lớn.
    * Thay đổi nhân sự chủ chốt (Chủ tịch, CEO), cơ cấu cổ đông lớn.
    * **Rủi ro:** Kiện tụng, tranh chấp, xử phạt vi phạm hành chính, sự cố môi trường, bắt bớ.
3.  **Thị trường & Nguyên liệu:**
    * Biến động giá nguyên vật liệu đầu vào quan trọng của ngành (ví dụ: giá dầu, giá điện, giá hạt nhựa, giá thép...).
    * Tin tức nóng về các đối thủ cạnh tranh lớn trong cùng ngành.

**Quy tắc cú pháp:**
* Luôn đặt tên công ty hoặc ngành hoặc mã cổ phiếu trong dấu ngoặc kép **"..."**.
* Sử dụng **Google dorking** giới hạn trong các nguồn uy tín:
    site:cafef.vn OR site:cafebiz.vn OR site:vietstock.vn OR site:vnexpress.net OR site:tuoitre.vn OR site:vneconomy.vn OR site:plo.vn OR site:thesaigontimes.vn OR site:diendandoanhnghiep.vn OR site:baodautu.vn
* Thêm "after:{start_date}" vào cuối mỗi từ khóa.

Chỉ trả về danh sách từ khóa, mỗi dòng 1 từ khóa.
"""

    try:
        response = model.generate_content(prompt)
        queries = [q.strip() for q in response.text.strip().split("\n") if q.strip()]

        print("\n=== Truy vấn AI tạo ra ===")
        for q in queries:
            print("-", q)

        return queries

    except Exception as e:
        print(f"Lỗi khi tạo query: {e}")
        return []


# ===========================================
# 2️⃣ FUNCTION TÌM KIẾM BẰNG SERPAPI
# ===========================================
def search_news_with_serpapi(queries):
    print("\nĐang tìm kiếm qua SerpAPI...")

    urls = []

    for query in queries:
        try:
            result = serp.search({
                "engine": "google",
                "q": query,
                "hl": "vi",
                "gl": "vn",
                "num": 10
            })

            organic = result.get("organic_results", [])
            for item in organic:
                url = item.get("link")
                if url:
                    urls.append(url)

        except Exception as e:
            print(f"Lỗi với query '{query}': {e}")

    unique_urls = list(dict.fromkeys(urls))

    return unique_urls


# ===========================================
# 3️⃣ FUNCTION CHÍNH — DÙNG 2 FUNCTION TRÊN
# ===========================================
def stock_news_agent_urls(stock_ticker, company_name, industry):
    print(f"Đang tìm kiếm tin tức về {company_name} ({stock_ticker})...")

    start_date = (datetime.date.today() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")

    # Step 1: Tạo query
    queries = generate_search_queries(stock_ticker, company_name, industry, start_date)
    if not queries:
        return []

    # Step 2: Tìm URL bằng SerpAPI
    urls = search_news_with_serpapi(queries)

    if not urls:
        print("Không tìm thấy tin tức.")
        return []

    return urls

def extract_name_from_profile(profile_text):
    # 1. Try to find the name enclosed in parentheses, which often follows the full name.
    # Pattern: \((.*?)\) looks for text inside the first pair of parentheses.
    match_parentheses = re.search(r'\((.*?)\)', profile_text)
    
    if match_parentheses:
        # Check if the content is a ticker (3-5 uppercase letters)
        ticker = match_parentheses.group(1).strip()
        if re.fullmatch(r'^[A-Z]{3,5}$', ticker):
             # If it's just the ticker (like NHH), we use it to find the name right before it.
             # Pattern: (.*?) finds the company name right before the ticker in parentheses.
            match_full_name = re.search(r'(.*?) \(' + re.escape(ticker) + r'\)', profile_text)
            if match_full_name:
                return match_full_name.group(1).strip()
        
        # If the content in parentheses is not just a ticker, it might be the official short name.
        return ticker

    # 2. If no parentheses are found or the above fails,
    # extract the text before the first comma (,), the first period (.), or "có tiền thân" (has predecessor)
    # This usually captures the main subject of the sentence.
    match_start = re.search(r'^(.+?)(?:, |\. | có tiền thân)', profile_text)
    if match_start:
        return match_start.group(1).strip()
        
    # 3. Fallback to the first sentence
    return profile_text.split('.')[0].strip()


# ======================
# 3. TEST FUNCTION
# ======================
def main(stock_ticker):
    company = Company(symbol=stock_ticker, source='VCI')

    company_profile = company.overview().company_profile.iloc[0]
    industry = company.overview().icb_name4[0]
    company_name = extract_name_from_profile(company_profile)

    results = stock_news_agent_urls(stock_ticker, company_name, industry)

    # 2. Construct the full file path
    today_date = date.today().strftime("%Y-%m-%d")
    filename = f"{stock_ticker}_{today_date}.txt"
    full_file_path = os.path.join("news", filename)

    # 3. Write the content to the file
    with open(full_file_path, 'w', encoding='utf-8') as f:
        f.write(f"--- DANH SÁCH URL TIN TỨC CHO MÃ {stock_ticker} ---\n")
        for i, url in enumerate(results, 1):
            f.write(f"{i}. {url}\n")

    print(f"✅ Kết quả đã được lưu tại: {full_file_path}")

if __name__ == '__main__':
    main("VHE")
