import google.generativeai as genai
from serpapi import Client
import datetime

# ====== CẤU HÌNH API ======
GENAI_API_KEY = "YOUR_API_KEY"
SERP_API_KEY = "YOUR_API_KEY"   # <<--- THAY API KEY CỦA BẠN

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Khởi tạo SerpAPI Client
serp = Client(api_key=SERP_API_KEY)

# ====== AI AGENT ======
def generate_search_queries(stock_ticker, company_name, start_date):
    prompt = f"""
    Dựa trên mã cổ phiếu "{stock_ticker}" và tên công ty "{company_name}",
    hãy tạo danh sách **5–10 từ khóa tìm kiếm hiệu quả** để tìm tin tức mới nhất.

    Nội dung gồm: biến động giá cổ phiếu, lý do tăng giảm, kết quả kinh doanh, dự án nổi bật,
    M&A, quản trị, vi phạm, kiện tụng, phát hành cổ phiếu, thay đổi lãnh đạo, cảnh báo rủi ro.

    Mỗi từ khóa phải có dấu "" bao quanh tên công ty hoặc mã cổ phiếu,
    áp dụng Google dorking và chỉ tìm trong các domain:
    site:cafef.vn OR site:cafebiz.vn OR site:vietstock.vn OR site:vnexpress.net OR
    site:tuoitre.vn OR site:vneconomy.vn OR site:plo.vn OR site:thesaigontimes.vn
    OR site:diendandoanhnghiep.vn OR site:baodautu.vn

    Thêm "after:{start_date}" vào cuối mỗi từ khóa.

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
def stock_news_agent_urls(stock_ticker, company_name):
    print(f"Đang tìm kiếm tin tức về {company_name} ({stock_ticker})...")

    start_date = (datetime.date.today() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")

    # Step 1: Tạo query
    queries = generate_search_queries(stock_ticker, company_name, start_date)
    if not queries:
        return []

    # Step 2: Tìm URL bằng SerpAPI
    urls = search_news_with_serpapi(queries)

    if not urls:
        print("Không tìm thấy tin tức.")
        return []

    return urls


# ======================
# 3. TEST FUNCTION
# ======================
stock_ticker = "CII"
company_name = "Công ty Cổ phần Đầu tư Hạ tầng Kỹ thuật TPHCM"

results = stock_news_agent_urls(stock_ticker, company_name)

print("\n--- DANH SÁCH URL TIN TỨC ---")
for i, url in enumerate(results, 1):
    print(f"{i}. {url}")
