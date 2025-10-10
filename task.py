# task.py
from crewai import Task
from vnstock import Quote, Finance,Company
from getData import load_price_data

# ------------------------------
# Helper functions to fetch data
# ------------------------------

def get_financials(symbol: str):
    """Get financial statements as dicts"""
    finance = Finance(symbol=symbol, source="VCI")
    return {
        "income": finance.income_statement(period="year", lang="vi").to_dict(orient="records"),
        "balance": finance.balance_sheet(period="year", lang="vi").to_dict(orient="records"),
        "cashflow": finance.cash_flow(period="year", lang="vi").to_dict(orient="records"),
        "ratios": finance.ratio(period="year", lang="vi").to_dict(orient="records")
    }


def get_news(symbol: str):
    """Get company news, events, and reports"""
    company = Company(symbol=symbol, source="VCI")
    return {
        "events": company.events().to_dict(orient="records"),
        "news": company.news().to_dict(orient="records"),
        "reports": company.reports().to_dict(orient="records")
    }

# ------------------------------
# Define CrewAI tasks
# ------------------------------

def create_tasks(symbol, agents):
    tasks = [Task(
        description=(
            f"Phân tích technical cho {symbol} dựa trên dữ liệu giá & volume:\n"
            f"{load_price_data(symbol)}"
        ),
        expected_output="Technical analysis với indicators và chart pattern.",
        agent=agents["technical"],
        )]
    return tasks

