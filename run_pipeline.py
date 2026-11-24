import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import briefing  # noqa: E402
import export_artefacts  # noqa: E402
import forecast  # noqa: E402
import ingest  # noqa: E402
import price_backtest  # noqa: E402
import quality  # noqa: E402

STEPS = [
    ("1. ingest latest data into DuckDB", ingest),
    ("2. data-quality checks and scorecard", quality),
    ("3. forecasting backtest", forecast),
    ("4. price-direction research backtest", price_backtest),
    ("5. LLM daily briefing with number check", briefing),
    ("6. export artefacts for the app", export_artefacts),
]


def main():
    for name, module in STEPS:
        print(f"\n===== {name} =====")
        module.main()
    print("\nPipeline complete. Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
