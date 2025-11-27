from vnstock import Company
import re

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

stock_ticker = "NHH"
company = Company(symbol=stock_ticker, source='VCI')
company_profile = company.overview().company_profile.iloc[0]
#company_name = extract_name_from_profile(company_profile)
industry = company.overview().icb_name4[0]

print(industry)