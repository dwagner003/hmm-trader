# src/trading/executor.py
"""IBKR executor for submitting MOC orders and querying positions.

Requires ibapi package (install manually from IBKR TWS API download).
"""
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    _HAS_IBAPI = True
except ImportError:
    _HAS_IBAPI = False
    logger.warning("ibapi not installed. IBKR trading will not be available.")


if _HAS_IBAPI:
    class IBKRApp(EWrapper, EClient):
        """Thin wrapper around ibapi for MOC orders and position queries."""

        def __init__(self):
            EClient.__init__(self, self)
            self.positions = {}  # type: Dict[str, float]
            self.cash = 0.0
            self.order_fills = {}  # type: Dict[int, dict]
            self._next_order_id = None  # type: Optional[int]
            self._position_done = threading.Event()
            self._account_done = threading.Event()

        def nextValidId(self, orderId):
            self._next_order_id = orderId

        def position(self, account, contract, pos, avgCost):
            self.positions[contract.symbol] = float(pos)

        def positionEnd(self):
            self._position_done.set()

        def updateAccountValue(self, key, val, currency, accountName):
            if key == "CashBalance" and currency == "USD":
                self.cash = float(val)

        def accountDownloadEnd(self, accountName):
            self._account_done.set()

        def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                        permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
            self.order_fills[orderId] = {
                "status": status, "filled": filled,
                "remaining": remaining, "avg_fill_price": avgFillPrice,
            }

        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
            if errorCode not in (2104, 2106, 2158):
                logger.error("IBKR Error %d: %s", errorCode, errorString)


def connect_ibkr(host, port, client_id):
    # type: (str, int, int) -> "IBKRApp"
    """Connect to IBKR gateway and return app instance."""
    if not _HAS_IBAPI:
        raise ImportError("ibapi is not installed. Install from IBKR TWS API download.")
    app = IBKRApp()
    app.connect(host, port, client_id)
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    timeout = 10
    while app._next_order_id is None and timeout > 0:
        time.sleep(0.5)
        timeout -= 0.5
    if app._next_order_id is None:
        raise ConnectionError("Failed to connect to IBKR gateway")
    return app


def get_positions_and_cash(app):
    # type: ("IBKRApp") -> Tuple[Dict[str, float], float]
    """Query current positions and cash balance."""
    app.positions = {}
    app.cash = 0.0
    app._position_done.clear()
    app._account_done.clear()
    app.reqPositions()
    app._position_done.wait(timeout=10)
    app.reqAccountUpdates(True, "")
    app._account_done.wait(timeout=10)
    app.reqAccountUpdates(False, "")
    return app.positions, app.cash


def submit_moc_orders(app, orders):
    # type: ("IBKRApp", List[dict]) -> List[dict]
    """Submit Market-on-Close orders to IBKR."""
    if not _HAS_IBAPI:
        raise ImportError("ibapi is not installed")
    submitted = []
    for order_spec in orders:
        contract = Contract()
        contract.symbol = order_spec["ticker"]
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        order = Order()
        order.action = order_spec["action"]
        order.totalQuantity = int(order_spec["shares"])
        order.orderType = "MOC"
        order.tif = "DAY"

        order_id = app._next_order_id
        app._next_order_id += 1

        app.placeOrder(order_id, contract, order)
        logger.info("Submitted MOC order #%d: %s %s %s",
                     order_id, order_spec["action"], order_spec["shares"], order_spec["ticker"])

        submitted.append({**order_spec, "order_id": order_id})
    return submitted


def check_fills(app, order_ids, timeout=60):
    # type: ("IBKRApp", List[int], int) -> Dict[int, dict]
    """Wait for order fills."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        all_filled = all(
            app.order_fills.get(oid, {}).get("status") == "Filled"
            for oid in order_ids
        )
        if all_filled:
            break
        time.sleep(1)
    return {oid: app.order_fills.get(oid, {"status": "Unknown"}) for oid in order_ids}
