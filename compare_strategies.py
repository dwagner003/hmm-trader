#!/usr/bin/env python3
"""Compare multiple HMM trading strategies and serve an interactive web dashboard.

All data displayed is generated locally from backtests - no untrusted content.
"""
import json
import os
import sys
import http.server
import socketserver
import webbrowser
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from src.config import load_config
from src.data.fetcher import backfill_history
from src.backtest.engine import run_backtest


def load_strategy_config(strategy_path):
    """Load a strategy YAML and merge with defaults."""
    with open("config/default.yaml") as f:
        base = yaml.safe_load(f)
    with open(strategy_path) as f:
        override = yaml.safe_load(f)
    for key, val in override.items():
        if isinstance(val, dict) and key in base and isinstance(base[key], dict):
            base[key].update(val)
        else:
            base[key] = val
    base["data_dir"] = os.environ.get("DATA_DIR", "./data")
    return base


def run_all_backtests():
    """Run backtests for all strategies and SPY benchmark."""
    strategies_dir = Path("config/strategies")
    strategy_files = sorted(strategies_dir.glob("*.yaml"))

    if not strategy_files:
        print("No strategy files found in config/strategies/")
        sys.exit(1)

    all_strategies = [("default", "config/default.yaml")] + [
        (f.stem, str(f)) for f in strategy_files
    ]

    print("Fetching 10 years of historical data...")
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "SHY"]
    prices = backfill_history(tickers, years=10)
    print("Data fetched: {} rows\n".format(len(prices)))

    # SPY benchmark
    print("Computing SPY buy-and-hold benchmark...")
    spy_prices = prices[prices["ticker"] == "SPY"].copy()
    spy_prices = spy_prices.sort_values("date").set_index("date")

    # Filter to backtest period
    spy_prices = spy_prices[spy_prices.index >= pd.Timestamp("2019-01-01")]
    spy_start = float(spy_prices["close"].iloc[0])
    spy_equity = (spy_prices["close"] / spy_start) * 10000
    spy_returns = spy_prices["close"].pct_change().dropna()
    spy_total = float(spy_prices["close"].iloc[-1] / spy_start) - 1.0
    spy_n_days = len(spy_prices)
    spy_ann = (1 + spy_total) ** (252 / max(spy_n_days, 1)) - 1.0
    spy_sharpe = 0.0
    if len(spy_returns) > 1 and float(spy_returns.std()) > 0:
        spy_sharpe = float((spy_returns.mean() / spy_returns.std()) * np.sqrt(252))
    spy_rolling_max = spy_prices["close"].cummax()
    spy_drawdowns = (spy_prices["close"] - spy_rolling_max) / spy_rolling_max
    spy_max_dd = float(spy_drawdowns.min())

    results = {}
    results["SPY Buy & Hold"] = {
        "total_return": spy_total,
        "annualized_return": spy_ann,
        "sharpe_ratio": spy_sharpe,
        "max_drawdown": spy_max_dd,
        "win_rate": 0,
        "total_trades": 0,
        "equity_dates": [str(d)[:10] for d in spy_equity.index],
        "equity_values": [round(float(v), 2) for v in spy_equity.values],
    }

    for name, path in all_strategies:
        print("Running backtest: {} ...".format(name))
        config = load_strategy_config(path)
        config["backtest"]["start_date"] = "2019-01-01"
        config["backtest"]["end_date"] = None
        config["backtest"]["initial_capital"] = 10000

        bt = run_backtest(config, prices)
        equity = bt["equity_curve"]
        results[name] = {
            "total_return": float(bt["total_return"]),
            "annualized_return": float(bt["annualized_return"]),
            "sharpe_ratio": float(bt["sharpe_ratio"]),
            "max_drawdown": float(bt["max_drawdown"]),
            "win_rate": float(bt.get("win_rate", 0)),
            "total_trades": len(bt["trade_log"]),
            "equity_dates": [str(d)[:10] for d in equity.index],
            "equity_values": [round(float(v), 2) for v in equity.values],
        }
        print("  Total: {:.2%}  Sharpe: {:.2f}  MaxDD: {:.2%}".format(
            bt["total_return"], bt["sharpe_ratio"], bt["max_drawdown"]))

    return results


def generate_dashboard(results):
    """Generate an HTML dashboard with interactive charts."""
    output_dir = Path("data/dashboard")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)

    colors = [
        "#2563eb", "#dc2626", "#16a34a", "#9333ea",
        "#ea580c", "#0891b2", "#4f46e5", "#be123c"
    ]

    # Build table rows safely using json-encoded data
    results_json = json.dumps(results)
    colors_json = json.dumps(colors)

    html_template = Path(__file__).parent / "dashboard_template.html"

    # Generate the dashboard HTML with embedded data
    dashboard_html = _build_dashboard_html(results_json, colors_json)

    dashboard_path = output_dir / "index.html"
    with open(dashboard_path, "w") as f:
        f.write(dashboard_html)

    return str(dashboard_path)


def _build_dashboard_html(results_json, colors_json):
    """Build the dashboard HTML string with embedded backtest data.

    All data is locally generated from backtests - no untrusted content.
    DOM elements are built using safe DOM APIs (createElement, textContent)
    in the embedded script.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HMM Trader - Strategy Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 24px;
        }
        h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; color: #f8fafc; }
        .subtitle { color: #94a3b8; margin-bottom: 32px; font-size: 14px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }
        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
        }
        .card h3 { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #f1f5f9; }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #334155;
            font-size: 14px;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: #94a3b8; }
        .metric-value { font-weight: 600; font-variant-numeric: tabular-nums; }
        .positive { color: #4ade80; }
        .negative { color: #f87171; }
        .chart-container {
            background: #1e293b;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid #334155;
            margin-bottom: 24px;
        }
        .chart-container h2 { font-size: 18px; font-weight: 600; margin-bottom: 16px; }
        canvas { width: 100% !important; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th {
            text-align: left; padding: 12px 16px;
            background: #1e293b; color: #94a3b8;
            font-weight: 600; border-bottom: 2px solid #334155;
        }
        td {
            padding: 12px 16px; border-bottom: 1px solid #1e293b;
            font-variant-numeric: tabular-nums;
        }
        tr:hover td { background: #1e293b; }
        .table-container {
            background: #1e293b; border-radius: 12px;
            padding: 24px; border: 1px solid #334155; overflow-x: auto;
        }
        .best { background: #166534; border-radius: 4px; padding: 2px 6px; }
    </style>
</head>
<body>
    <h1>HMM Trader - Strategy Comparison</h1>
    <p class="subtitle">Backtest period: 2019 - 2026 | Initial capital: $10,000 | Walk-forward HMM regime detection</p>

    <div class="chart-container">
        <h2>Equity Curves</h2>
        <canvas id="equityChart" height="100"></canvas>
    </div>

    <div class="chart-container">
        <h2>Drawdown</h2>
        <canvas id="drawdownChart" height="80"></canvas>
    </div>

    <div class="table-container" style="margin-bottom: 24px;">
        <h2 style="font-size: 18px; font-weight: 600; margin-bottom: 16px;">Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Total Return</th>
                    <th>Annual Return</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                </tr>
            </thead>
            <tbody id="summaryTable"></tbody>
        </table>
    </div>

    <div class="grid" id="strategyCards"></div>

    <script>
    // All data is generated locally from backtests - safe to embed
    const results = """ + results_json + """;
    const colors = """ + colors_json + """;
    const names = Object.keys(results);

    // --- Summary Table (built with safe DOM methods) ---
    const tbody = document.getElementById('summaryTable');
    let bestReturn = -Infinity, bestSharpe = -Infinity, bestDD = -Infinity;
    names.forEach(function(n) {
        var r = results[n];
        if (r.total_return > bestReturn) bestReturn = r.total_return;
        if (r.sharpe_ratio > bestSharpe) bestSharpe = r.sharpe_ratio;
        if (r.max_drawdown > bestDD) bestDD = r.max_drawdown;
    });

    names.forEach(function(name, i) {
        var r = results[name];
        var tr = document.createElement('tr');

        // Strategy name cell
        var tdName = document.createElement('td');
        var dot = document.createElement('span');
        dot.style.color = colors[i % colors.length];
        dot.style.fontWeight = '600';
        dot.textContent = '\u25cf ';
        tdName.appendChild(dot);
        tdName.appendChild(document.createTextNode(name));
        tr.appendChild(tdName);

        // Total return
        var tdRet = document.createElement('td');
        tdRet.className = r.total_return >= 0 ? 'positive' : 'negative';
        tdRet.textContent = (r.total_return * 100).toFixed(2) + '%';
        if (r.total_return === bestReturn) tdRet.classList.add('best');
        tr.appendChild(tdRet);

        // Annual return
        var tdAnn = document.createElement('td');
        tdAnn.className = r.annualized_return >= 0 ? 'positive' : 'negative';
        tdAnn.textContent = (r.annualized_return * 100).toFixed(2) + '%';
        tr.appendChild(tdAnn);

        // Sharpe
        var tdSharpe = document.createElement('td');
        tdSharpe.textContent = r.sharpe_ratio.toFixed(2);
        if (r.sharpe_ratio === bestSharpe) tdSharpe.classList.add('best');
        tr.appendChild(tdSharpe);

        // Max drawdown
        var tdDD = document.createElement('td');
        tdDD.className = 'negative';
        tdDD.textContent = (r.max_drawdown * 100).toFixed(2) + '%';
        if (r.max_drawdown === bestDD) tdDD.classList.add('best');
        tr.appendChild(tdDD);

        // Win rate
        var tdWin = document.createElement('td');
        tdWin.textContent = (r.win_rate * 100).toFixed(1) + '%';
        tr.appendChild(tdWin);

        // Trades
        var tdTrades = document.createElement('td');
        tdTrades.textContent = r.total_trades.toString();
        tr.appendChild(tdTrades);

        tbody.appendChild(tr);
    });

    // --- Equity Chart ---
    var equityCtx = document.getElementById('equityChart').getContext('2d');
    var equityDatasets = names.map(function(name, i) {
        return {
            label: name,
            data: results[name].equity_dates.map(function(d, j) {
                return { x: d, y: results[name].equity_values[j] };
            }),
            borderColor: colors[i % colors.length],
            backgroundColor: 'transparent',
            borderWidth: name === 'SPY Buy & Hold' ? 3 : 2,
            borderDash: name === 'SPY Buy & Hold' ? [6, 3] : [],
            pointRadius: 0,
            tension: 0.1
        };
    });

    new Chart(equityCtx, {
        type: 'line',
        data: { datasets: equityDatasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'category',
                    ticks: { maxTicksLimit: 12, color: '#64748b' },
                    grid: { color: '#1e293b' }
                },
                y: {
                    ticks: {
                        color: '#64748b',
                        callback: function(v) { return '$' + v.toLocaleString(); }
                    },
                    grid: { color: '#1e293b' }
                }
            },
            plugins: {
                legend: { labels: { color: '#e2e8f0', usePointStyle: true } },
                tooltip: {
                    callbacks: {
                        label: function(ctx) { return ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString(); }
                    }
                }
            }
        }
    });

    // --- Drawdown Chart ---
    var ddCtx = document.getElementById('drawdownChart').getContext('2d');
    var ddDatasets = names.map(function(name, i) {
        var vals = results[name].equity_values;
        var peak = vals[0];
        var dd = vals.map(function(v) {
            if (v > peak) peak = v;
            return ((v - peak) / peak) * 100;
        });
        return {
            label: name,
            data: results[name].equity_dates.map(function(d, j) {
                return { x: d, y: dd[j] };
            }),
            borderColor: colors[i % colors.length],
            backgroundColor: 'transparent',
            borderWidth: name === 'SPY Buy & Hold' ? 3 : 1.5,
            borderDash: name === 'SPY Buy & Hold' ? [6, 3] : [],
            pointRadius: 0,
            tension: 0.1
        };
    });

    new Chart(ddCtx, {
        type: 'line',
        data: { datasets: ddDatasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'category',
                    ticks: { maxTicksLimit: 12, color: '#64748b' },
                    grid: { color: '#1e293b' }
                },
                y: {
                    ticks: {
                        color: '#64748b',
                        callback: function(v) { return v.toFixed(0) + '%'; }
                    },
                    grid: { color: '#1e293b' }
                }
            },
            plugins: {
                legend: { labels: { color: '#e2e8f0', usePointStyle: true } },
                tooltip: {
                    callbacks: {
                        label: function(ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + '%'; }
                    }
                }
            }
        }
    });

    // --- Strategy Cards (built with safe DOM methods) ---
    var cardsDiv = document.getElementById('strategyCards');
    names.forEach(function(name, i) {
        if (name === 'SPY Buy & Hold') return;
        var r = results[name];
        var card = document.createElement('div');
        card.className = 'card';

        var h3 = document.createElement('h3');
        var dot = document.createElement('span');
        dot.style.color = colors[i % colors.length];
        dot.textContent = '\u25cf ';
        h3.appendChild(dot);
        h3.appendChild(document.createTextNode(name));
        card.appendChild(h3);

        var metrics = [
            ['Total Return', (r.total_return * 100).toFixed(2) + '%', r.total_return >= 0],
            ['Annual Return', (r.annualized_return * 100).toFixed(2) + '%', r.annualized_return >= 0],
            ['Sharpe Ratio', r.sharpe_ratio.toFixed(2), true],
            ['Max Drawdown', (r.max_drawdown * 100).toFixed(2) + '%', false],
            ['Win Rate', (r.win_rate * 100).toFixed(1) + '%', true],
            ['Total Trades', r.total_trades.toString(), true]
        ];

        metrics.forEach(function(m) {
            var row = document.createElement('div');
            row.className = 'metric-row';
            var label = document.createElement('span');
            label.className = 'metric-label';
            label.textContent = m[0];
            var value = document.createElement('span');
            value.className = 'metric-value ' + (m[2] ? 'positive' : 'negative');
            if (m[0] === 'Sharpe Ratio' || m[0] === 'Total Trades' || m[0] === 'Win Rate') {
                value.className = 'metric-value';
            }
            value.textContent = m[1];
            row.appendChild(label);
            row.appendChild(value);
            card.appendChild(row);
        });

        cardsDiv.appendChild(card);
    });
    </script>
</body>
</html>"""


def serve_dashboard(path, port=8765):
    """Serve the dashboard on a local HTTP server."""
    directory = str(Path(path).parent)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, format, *args):
            pass

    with socketserver.TCPServer(("", port), Handler) as httpd:
        url = "http://localhost:{}".format(port)
        print("\nDashboard ready at: {}".format(url))
        print("Press Ctrl+C to stop.\n")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    results = run_all_backtests()
    path = generate_dashboard(results)
    print("\nDashboard saved to: {}".format(path))
    serve_dashboard(path)
