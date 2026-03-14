import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List


def send_alert(subject, body, smtp_host, smtp_port, smtp_user, smtp_password, to_email):
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = "[HMM Trader] {}".format(subject)
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def format_rebalance_email(regime_probs, trades, portfolio_value):
    # type: (Dict[str, float], List[dict], float) -> str
    lines = ["Daily Rebalance Executed", "=" * 40, ""]
    lines.append("Regime Probabilities:")
    for regime, prob in sorted(regime_probs.items()):
        lines.append("  {:>10s}: {:.1%}".format(regime, prob))
    lines.append("\nPortfolio Value: ${:,.2f}".format(portfolio_value))
    lines.append("\nTrades ({}):".format(len(trades)))
    if trades:
        for t in trades:
            lines.append("  {:>4s} {:>8.2f} {} @ ${:>10.2f}".format(
                t["action"], t.get("shares", 0), t["ticker"], t.get("price", 0)))
    else:
        lines.append("  No trades executed.")
    return "\n".join(lines)


def format_error_email(error_msg, context):
    # type: (str, str) -> str
    return "\n".join([
        "SYSTEM ERROR", "=" * 40, "",
        "Context: {}".format(context), "", "Error:", error_msg, "",
        "Please check the system logs for details.",
    ])


def format_circuit_breaker_email(current_value, peak_value, drawdown_pct):
    # type: (float, float, float) -> str
    return "\n".join([
        "CIRCUIT BREAKER TRIGGERED", "=" * 40, "",
        "Peak Value:    ${:,.2f}".format(peak_value),
        "Current Value: ${:,.2f}".format(current_value),
        "Drawdown:      {:.1%}".format(drawdown_pct), "",
        "All positions have been liquidated.",
        "Manual restart required: python -m src.main reset-circuit-breaker",
    ])


def format_weekly_summary_email(week_return, total_value, regime, regime_probs, holdings):
    # type: (float, float, str, Dict[str, float], Dict[str, float]) -> str
    lines = ["Weekly Summary", "=" * 40, "",
        "Portfolio Value: ${:,.2f}".format(total_value),
        "Week Return:     {:.2%}".format(week_return),
        "Current Regime:  {}".format(regime), "",
        "Regime Probabilities:"]
    for r, p in sorted(regime_probs.items()):
        lines.append("  {:>10s}: {:.1%}".format(r, p))
    lines.append("")
    lines.append("Holdings:")
    for ticker, shares in sorted(holdings.items()):
        lines.append("  {}: {:.2f} shares".format(ticker, shares))
    return "\n".join(lines)
