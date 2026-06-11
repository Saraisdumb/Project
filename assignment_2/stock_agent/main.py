import sys
from stock_agent.graph import stock_agent
from stock_agent.state import StockAgentState


def run(ticker: str) -> None:
    state = StockAgentState(
        ticker=ticker,
        raw_data=None,
        indicators=None,
        recommendation=None,
        report=None,
        error=None,
    )
    final = stock_agent.invoke(state)
    print(final["report"])


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run(symbol)
