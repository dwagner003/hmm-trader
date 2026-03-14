# src/backtest/report.py
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def print_summary(results: dict):
    """Print backtest summary metrics to console."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:      {results['total_return']:>10.2%}")
    print(f"  Annualized Return: {results['annualized_return']:>10.2%}")
    print(f"  Sharpe Ratio:      {results['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown:      {results['max_drawdown']:>10.2%}")
    print(f"  Win Rate:          {results.get('win_rate', 0):>10.2%}")
    print(f"  Total Trades:      {len(results['trade_log']):>10d}")
    print("=" * 60 + "\n")


def generate_charts(results: dict, output_dir: str) -> List[str]:
    """Generate charts and HTML report. Returns list of output file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []  # type: List[str]

    equity = results["equity_curve"]
    regime_df = results["regime_history"]

    if equity.empty:
        return paths

    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, label="HMM Strategy", linewidth=1.5)
    ax.axhline(y=equity.iloc[0], color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = str(out / "equity_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 2. Drawdown chart
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color="red")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = str(out / "drawdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 3. Regime timeline
    if not regime_df.empty:
        regime_colors = {"bull": "green", "sideways": "gold", "bear": "red"}
        fig, ax = plt.subplots(figsize=(12, 3))
        dates = pd.to_datetime(regime_df["date"])
        colors = [regime_colors.get(r, "gray") for r in regime_df["regime"]]
        ax.bar(dates, [1] * len(dates), color=colors, width=1.5)
        ax.set_title("Detected Market Regimes")
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=r) for r, c in regime_colors.items()]
        ax.legend(handles=legend_elements, loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = str(out / "regime_timeline.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # Generate HTML report
    if paths:
        html_path = str(out / "report.html")
        html_lines = [
            "<html><head><title>HMM Trader Backtest Report</title></head><body>",
            "<h1>Backtest Report</h1>",
            "<p>Total Return: {:.2%}</p>".format(results["total_return"]),
            "<p>Annualized Return: {:.2%}</p>".format(results["annualized_return"]),
            "<p>Sharpe Ratio: {:.2f}</p>".format(results["sharpe_ratio"]),
            "<p>Max Drawdown: {:.2%}</p>".format(results["max_drawdown"]),
            "<p>Win Rate: {:.2%}</p>".format(results.get("win_rate", 0)),
            "<p>Total Trades: {}</p>".format(len(results["trade_log"])),
        ]
        for p in paths:
            fname = Path(p).name
            html_lines.append('<img src="{}" style="max-width:100%"><br>'.format(fname))
        html_lines.append("</body></html>")
        with open(html_path, "w") as f:
            f.write("\n".join(html_lines))
        paths.append(html_path)

    return paths
