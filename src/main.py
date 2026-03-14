"""HMM Trader - main entry point.

Commands:
    run                    Run the daily rebalance job
    backtest               Run a backtest against historical data
    reset-circuit-breaker  Reset the circuit breaker after manual review
"""
import argparse
import json
import logging
import os
import sys
from datetime import date

from src.config import load_config
from src.data.fetcher import fetch_prices, backfill_history
from src.data.features import compute_features
from src.data.store import DataStore
from src.model.hmm import RegimeHMM
from src.model.allocator import compute_target_weights
from src.trading.portfolio import compute_current_weights, compute_orders
from src.trading.risk import (
    check_position_limits, check_rebalance_needed, check_drawdown,
    check_turnover, is_trading_day,
)
from src.alerts.email import (
    send_alert, format_rebalance_email, format_error_email,
    format_circuit_breaker_email, format_weekly_summary_email,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("hmm_trader")


def _send_email(config, subject, body):
    """Send an email alert if SMTP is configured."""
    smtp_host = os.environ.get("SMTP_HOST")
    if not smtp_host:
        logger.warning("SMTP not configured, skipping email alert")
        return
    try:
        send_alert(
            subject=subject, body=body, smtp_host=smtp_host,
            smtp_port=int(os.environ.get("SMTP_PORT", 587)),
            smtp_user=os.environ["SMTP_USER"],
            smtp_password=os.environ["SMTP_PASSWORD"],
            to_email=os.environ["ALERT_EMAIL_TO"],
        )
    except Exception as e:
        logger.error("Failed to send email: %s", e)


def cmd_run(config):
    """Run the daily rebalance job."""
    today = date.today()

    if not is_trading_day(today):
        logger.info("Market closed on %s, skipping rebalance", today)
        return

    store = DataStore(config["data_dir"])

    if store.get_circuit_breaker_active():
        logger.warning("Circuit breaker is active. Run reset-circuit-breaker to re-enable trading.")
        return

    etf_universe = config["etf_universe"]

    # Check if historical data exists; backfill on first run
    existing_prices = store.load_prices()
    if existing_prices.empty:
        logger.info("No historical data found. Backfilling 10 years...")
        history = backfill_history(etf_universe, years=10)
        if history.empty:
            _send_email(config, "Error: Backfill Failed",
                       format_error_email("Historical data fetch returned empty", "Backfill"))
            return
        store.save_prices(history)

    # Fetch latest prices
    logger.info("Fetching latest prices...")
    prices_df = fetch_prices(etf_universe, period="5d")
    if prices_df.empty:
        _send_email(config, "Error: No Price Data",
                   format_error_email("Price fetch returned empty", "Daily run"))
        return

    store.save_prices(prices_df)

    # Compute features from full price history
    all_prices = store.load_prices()
    features = compute_features(all_prices)

    # Load or train model
    model = RegimeHMM(n_states=config["hmm"]["n_states"])
    try:
        model.load(config["data_dir"])
        logger.info("Loaded existing model")
    except FileNotFoundError:
        logger.info("No existing model found, training from scratch...")

    # Retrain if needed (first trading day of month or no model)
    training_window = config["hmm"]["training_window_years"] * 252
    last_retrain_month = store.get_system_state("last_retrain_month")
    current_month_key = "{}-{:02d}".format(today.year, today.month)
    needs_retrain = not model.is_trained or last_retrain_month != current_month_key
    if needs_retrain:
        if len(features) >= training_window:
            train_data = features.iloc[-training_window:]
            model.train(train_data)
            model.save(config["data_dir"])
            store.set_system_state("last_retrain_month", current_month_key)
            logger.info("Model trained and saved")
        else:
            logger.error("Not enough data to train: %d < %d", len(features), training_window)
            return

    # Predict regime
    window = features.iloc[-training_window:]
    regime_probs = model.predict(window)
    dominant = max(regime_probs, key=regime_probs.get)
    logger.info("Regime: %s | Probs: %s", dominant,
                json.dumps({k: "{:.2%}".format(v) for k, v in regime_probs.items()}))

    store.save_prediction(
        date=str(today), bull_prob=regime_probs["bull"],
        sideways_prob=regime_probs["sideways"], bear_prob=regime_probs["bear"],
        regime=dominant,
    )

    # Compute target weights
    target_weights = compute_target_weights(regime_probs, config)
    target_weights = check_position_limits(target_weights, config["risk"]["max_position_pct"])

    if config["mode"] == "backtest":
        logger.info("Backtest mode - skipping trade execution")
        return

    # Connect to IBKR (lazy import since ibapi may not be installed)
    from src.trading.executor import connect_ibkr, get_positions_and_cash, submit_moc_orders

    ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.environ.get("IBKR_PORT", 7497))
    ibkr_client_id = int(os.environ.get("IBKR_CLIENT_ID", 1))

    try:
        app = connect_ibkr(ibkr_host, ibkr_port, ibkr_client_id)
    except (ConnectionError, ImportError) as e:
        _send_email(config, "Error: IBKR Connection Failed",
                   format_error_email(str(e), "IBKR connect"))
        return

    try:
        holdings, cash = get_positions_and_cash(app)

        # Get latest prices as dict
        latest_prices = {}
        for t in etf_universe:
            t_prices = prices_df[prices_df["ticker"] == t]
            if not t_prices.empty:
                latest_prices[t] = float(t_prices.iloc[-1]["close"])

        total_value = cash + sum(
            holdings.get(t, 0) * latest_prices.get(t, 0) for t in etf_universe
        )

        # Check drawdown
        peak = store.get_peak_value() or total_value
        if total_value > peak:
            peak = total_value
            store.set_peak_value(peak)

        if check_drawdown(total_value, peak, config["risk"]["drawdown_circuit_breaker"]):
            logger.warning("CIRCUIT BREAKER TRIGGERED")
            store.set_circuit_breaker_active(True)
            drawdown_pct = (peak - total_value) / peak
            _send_email(config, "CIRCUIT BREAKER TRIGGERED",
                       format_circuit_breaker_email(total_value, peak, drawdown_pct))
            # Liquidate all positions
            liquidation_orders = []
            for ticker, shares in holdings.items():
                if shares > 0 and ticker in latest_prices:
                    liquidation_orders.append({
                        "ticker": ticker, "action": "SELL",
                        "shares": shares, "price": latest_prices[ticker],
                    })
            if liquidation_orders:
                submit_moc_orders(app, liquidation_orders)
                logger.info("Liquidated %d positions", len(liquidation_orders))
                for order in liquidation_orders:
                    store.save_trade(
                        date=str(today), ticker=order["ticker"], action="SELL",
                        shares=order["shares"], price=order["price"], cost=1.0,
                    )
            return

        # Compute current weights
        current_weights = compute_current_weights(holdings, latest_prices, cash)

        if not check_rebalance_needed(current_weights, target_weights,
                                       config["risk"]["rebalance_threshold_pp"]):
            logger.info("No rebalance needed - all positions within threshold")
            _send_email(config, "No Rebalance Needed",
                       format_rebalance_email(regime_probs, [], total_value))
            return

        target_weights = check_turnover(current_weights, target_weights,
                                        config["risk"]["max_daily_turnover"])

        orders = compute_orders(holdings, target_weights, latest_prices, total_value)
        if orders:
            submitted = submit_moc_orders(app, orders)
            logger.info("Submitted %d orders", len(submitted))
            _send_email(config, "Rebalance Executed",
                       format_rebalance_email(regime_probs, orders, total_value))
            for order in orders:
                store.save_trade(
                    date=str(today), ticker=order["ticker"], action=order["action"],
                    shares=order["shares"], price=order["price"], cost=1.0,
                )

        store.save_snapshot(
            date=str(today), total_value=total_value, cash=cash, holdings=holdings,
        )

        # Weekly summary on Fridays
        if today.weekday() == 4:
            snapshots = store.load_snapshots()
            if len(snapshots) >= 5:
                week_ago_value = snapshots.iloc[-5]["total_value"]
                week_return = (total_value - week_ago_value) / week_ago_value
            else:
                week_return = 0.0
            _send_email(config, "Weekly Summary",
                       format_weekly_summary_email(week_return, total_value, dominant,
                                                    regime_probs, holdings))
    finally:
        app.disconnect()


def cmd_backtest(config):
    """Run a backtest against historical data."""
    from src.backtest.engine import run_backtest
    from src.backtest.report import print_summary, generate_charts

    etf_universe = config["etf_universe"]
    logger.info("Fetching historical data for backtest...")
    prices = backfill_history(etf_universe, years=10)

    logger.info("Running backtest...")
    results = run_backtest(config, prices)
    print_summary(results)

    output_dir = os.path.join(config["data_dir"], "backtest_results")
    charts = generate_charts(results, output_dir)
    if charts:
        logger.info("Charts saved to %s", output_dir)


def cmd_reset_circuit_breaker(config):
    """Reset the circuit breaker after manual review."""
    store = DataStore(config["data_dir"])
    store.set_circuit_breaker_active(False)
    store.set_peak_value(0.0)
    logger.info("Circuit breaker reset. Trading will resume on next run.")


def main():
    parser = argparse.ArgumentParser(description="HMM Stock Trader")
    parser.add_argument(
        "command", choices=["run", "backtest", "reset-circuit-breaker"],
        help="Command to execute",
    )
    parser.add_argument("--config", default=None, help="Config override (e.g. 'paper' or 'live')")
    args = parser.parse_args()

    config = load_config(config_dir="config", override=args.config)

    commands = {
        "run": cmd_run,
        "backtest": cmd_backtest,
        "reset-circuit-breaker": cmd_reset_circuit_breaker,
    }

    try:
        commands[args.command](config)
    except Exception as e:
        logger.exception("Fatal error in %s", args.command)
        _send_email(config, "Fatal Error: {}".format(args.command),
                   format_error_email(str(e), args.command))
        sys.exit(1)


if __name__ == "__main__":
    main()
