import google.generativeai as genai
from googlesearch import search
import os
import datetime

# Cấu hình khóa API của bạn
genai.configure(api_key="AIzaSyDvebfu5lTsbyza9G7IhSrEDGKImDOtUFg")

# Khởi tạo mô hình Gemini
model = genai.GenerativeModel('gemini-1.5-flash')

def stock_news_agent_urls(stock_ticker, company_name):
    """
    AI Agent tìm kiếm và trả về danh sách URL tin tức về mã cổ phiếu trong 3 tháng gần nhất.
    """
    print(f"Đang tìm kiếm tin tức về {company_name} ({stock_ticker})...")

    # Tính toán ngày bắt đầu 3 tháng trước
    three_months_ago = datetime.date.today() - datetime.timedelta(days=90)
    start_date = three_months_ago.strftime("%Y-%m-%d")

    # Bước 1: Dùng Gemini để tạo các truy vấn tìm kiếm
    query_creation_prompt = f"""
    Dựa trên mã cổ phiếu "{stock_ticker}" và tên công ty "{company_name}", hãy tạo một danh sách các từ khóa tìm kiếm hiệu quả và đa dạng để tìm kiếm tin tức mới nhất về công ty này, bao gồm cả lý do tăng giảm giá cổ phiếu, hoạt động kinh doanh, và các dự án nổi bật. 
    
    Hãy thêm toán tử tìm kiếm 'after:{start_date}' vào cuối mỗi từ khóa để chỉ tìm kiếm các bài viết sau ngày này.
    
    Chỉ trả về danh sách các từ khóa, mỗi từ khóa trên một dòng. Không thêm bất kỳ lời giải thích nào.
    """
    try:
        response = model.generate_content(query_creation_prompt)
        search_queries = response.text.strip().split('\n')
        print("Đã tạo các từ khóa tìm kiếm:")
        for query in search_queries:
            print(f"- {query.strip()}")
    except Exception as e:
        print(f"Lỗi khi tạo truy vấn: {e}")
        return "Xin lỗi, không thể tạo truy vấn tìm kiếm."
        
    # Bước 2: Thực hiện tìm kiếm và lấy URL
    all_urls = []
    print("\nĐang tìm kiếm trên Google...")
    for query in search_queries:
        try:
            urls = list(search(query.strip(), num_results=10, lang="vi"))
            all_urls.extend(urls)
        except Exception as e:
            print(f"Lỗi khi tìm kiếm với truy vấn '{query.strip()}': {e}")
            continue
    
    # Loại bỏ các URL trùng lặp
    unique_urls = list(dict.fromkeys(all_urls))
    
    if not unique_urls:
        return "Không tìm thấy tin tức nào liên quan."

    return unique_urls

# Ví dụ sử dụng AI Agent mới
stock_ticker_input = "CII"
company_name_input = "Công ty Cổ phần Đầu tư Hạ tầng Kỹ thuật TPHCM"
urls = stock_news_agent_urls(stock_ticker_input, company_name_input)

print("\n--- Danh sách các đường link tin tức trong 3 tháng gần nhất ---")
if isinstance(urls, list):
    for i, url in enumerate(urls):
        print(f"{i+1}. {url}")
else:
    print(urls)