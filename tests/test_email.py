import pytest
from unittest.mock import patch, MagicMock
from src.alerts.email import send_alert, format_rebalance_email, format_error_email


def test_format_rebalance_email():
    body = format_rebalance_email(
        regime_probs={"bull": 0.7, "sideways": 0.2, "bear": 0.1},
        trades=[{"ticker": "SPY", "action": "BUY", "shares": 5, "price": 450.0}],
        portfolio_value=10000.0,
    )
    assert "bull" in body.lower()
    assert "SPY" in body
    assert "BUY" in body


def test_format_error_email():
    body = format_error_email(error_msg="Connection timeout", context="Daily rebalance job")
    assert "Connection timeout" in body
    assert "Daily rebalance" in body


@patch("src.alerts.email.smtplib.SMTP")
def test_send_alert_connects_and_sends(mock_smtp_class):
    mock_smtp = MagicMock()
    mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)
    send_alert(
        subject="Test", body="Test body",
        smtp_host="smtp.test.com", smtp_port=587,
        smtp_user="user@test.com", smtp_password="password", to_email="r@test.com",
    )
    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("user@test.com", "password")
    mock_smtp.send_message.assert_called_once()
