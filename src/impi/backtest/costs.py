"""Indian delivery-trade cost model (config/costs.yaml). Returns rupee costs for one order."""
from __future__ import annotations

from impi.common import config


def slippage_bps(bucket) -> float:
    s = config()["costs"]["slippage_bps"]
    return float(s.get(bucket, s["small"])) if isinstance(bucket, str) else float(s["small"])


def buy_cost(notional: float) -> float:
    c = config()["costs"]
    brok = c["brokerage_per_order"]
    exch = notional * c["exchange_txn"]
    sebi = notional * c["sebi_fee"]
    return (brok + notional * c["stt_buy"] + exch + sebi + notional * c["ipft"]
            + notional * c["stamp_duty_buy"] + c["gst"] * (brok + exch + sebi))


def sell_cost(notional: float) -> float:
    c = config()["costs"]
    brok = c["brokerage_per_order"]
    exch = notional * c["exchange_txn"]
    sebi = notional * c["sebi_fee"]
    return (brok + notional * c["stt_sell"] + exch + sebi + notional * c["ipft"]
            + c["gst"] * (brok + exch + sebi) + c["dp_charge_per_sell"])


def round_trip_pct(notional: float, bucket="large") -> float:
    """Total round-trip cost as a fraction of notional, including slippage on both legs."""
    return (buy_cost(notional) + sell_cost(notional)) / notional + 2 * slippage_bps(bucket) / 1e4
