# LangGraph Assessment

Two LangGraph-based AI agents built as part of a practical assessment.

## Repository Structure

```
langgraph-assessment/
├── assignment_1/
│   ├── weather_agent_debug.ipynb   # Bug report + proof of fix
│   └── weather_agent/              # Fixed weather agent source
├── assignment_2/
│   ├── stock_agent.ipynb           # Demo + unit tests notebook
│   └── stock_agent/                # Stock analysis agent source
│       ├── state.py
│       ├── nodes.py
│       ├── graph.py
│       └── main.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Assignment 1 — Weather Agent Debugging

The original weather agent had **8 bugs** across 4 files that prevented it from running at all.

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `main.py` | `"_main_"` typo | `"__main__"` |
| 2 | `graph.py` | Missing edge to `fetch_weather_data` | Added correct edge chain |
| 3 | `graph.py` | Graph never compiled | `builder.compile()` |
| 4 | `nodes.py` | Location data overwritten with `{}` | Store actual API response |
| 5 | `nodes.py` | `wind_unit` variable commented out | Uncommented |
| 6 | `nodes.py` | `wind_unit` as string literal in f-string | `{wind_unit}` |
| 7 | `config.py` | Duplicate field blanked API URL; malformed TEMP_MIN | Removed duplicate; fixed type |
| 8 | `helper_functions.py` | Truthy check `if temp_celsius:` | `if temp_celsius < config.TEMP_MIN:` |

See `assignment_1/weather_agent_debug.ipynb` for the full walkthrough.

## Assignment 2 — Stock Market Analysis Agent

A LangGraph agent that analyses any stock ticker and generates a BUY / HOLD / SELL recommendation.

### Graph Flow

```
START -> validate_ticker -> fetch_stock_data -> calculate_indicators
      -> generate_recommendation -> format_report -> END
```

### Technical Indicators

- **SMA 10** — 10-day Simple Moving Average
- **SMA 20** — 20-day Simple Moving Average
- **RSI 14** — 14-day Relative Strength Index

### Recommendation Logic

| Condition | Recommendation |
|-----------|---------------|
| SMA10 > SMA20 AND RSI < 70 | **BUY** |
| SMA10 < SMA20 AND RSI > 30 | **SELL** |
| Otherwise | **HOLD** |

### Run from command line

```bash
cd assignment_2
python -c "import sys; sys.path.insert(0,'.'); from stock_agent.main import run; run('AAPL')"
```

### Run in Jupyter

Open `assignment_2/stock_agent.ipynb` and run all cells.

### Error Handling

- Invalid ticker symbols (non-alpha, too long) are caught at `validate_ticker`
- Network errors and delisted tickers are caught at `fetch_stock_data`
- All errors propagate cleanly to a formatted error report
