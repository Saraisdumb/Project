from typing import TypedDict, Optional, Any, Dict


class StockAgentState(TypedDict):
    ticker: str
    raw_data: Optional[Any]
    indicators: Optional[Dict[str, float]]
    recommendation: Optional[str]
    report: Optional[str]
    error: Optional[str]
