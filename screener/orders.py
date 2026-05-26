from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any

from .breakout import BreakoutSignal

TRAILING_CALLBACK_MIN_PCT = Decimal("0.1")
TRAILING_CALLBACK_MAX_PCT = Decimal("10")
TRAILING_CALLBACK_STEP_PCT = Decimal("0.1")


class OrderPlanError(ValueError):
    """Raised when a signal cannot be turned into a valid futures order."""


@dataclass(frozen=True)
class TradingRule:
    symbol: str
    price_tick_size: Decimal
    quantity_step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal


@dataclass(frozen=True)
class ConditionalOrderPlan:
    role: str
    symbol: str
    interval: str
    signal_side: str
    binance_side: str
    order_type: str
    trigger_price: str
    limit_price: str
    quantity: str
    requested_notional: str
    estimated_notional: str
    client_order_id: str
    payload: dict[str, str]


def trading_rules_from_exchange_info(exchange_info: dict[str, Any]) -> dict[str, TradingRule]:
    rules: dict[str, TradingRule] = {}
    for raw_symbol in exchange_info.get("symbols", []):
        symbol = raw_symbol.get("symbol")
        if not symbol:
            continue

        filters = {item.get("filterType"): item for item in raw_symbol.get("filters", [])}
        price_filter = filters.get("PRICE_FILTER", {})
        lot_filter = filters.get("LOT_SIZE", {})
        market_lot_filter = filters.get("MARKET_LOT_SIZE", {})
        notional_filter = filters.get("MIN_NOTIONAL", {})

        quantity_step_size = _decimal(market_lot_filter.get("stepSize")) or _decimal(lot_filter.get("stepSize"))
        min_qty = _decimal(market_lot_filter.get("minQty")) or _decimal(lot_filter.get("minQty"))
        max_qty = _decimal(market_lot_filter.get("maxQty")) or _decimal(lot_filter.get("maxQty"))

        rules[symbol] = TradingRule(
            symbol=symbol,
            price_tick_size=_decimal(price_filter.get("tickSize")) or Decimal("0.00000001"),
            quantity_step_size=quantity_step_size or Decimal("0.00000001"),
            min_qty=min_qty or Decimal("0"),
            max_qty=max_qty or Decimal("0"),
            min_notional=_decimal(notional_filter.get("notional")) or Decimal("0"),
        )
    return rules


def build_entry_order_plan(
    signal: BreakoutSignal,
    rule: TradingRule,
    requested_notional: float,
    client_order_id: str,
    working_type: str,
    price_protect: bool,
    hedge_mode: bool,
    entry_mode: str = "STOP_MARKET",
    entry_pullback_pct: float = 0.0,
) -> ConditionalOrderPlan:
    if requested_notional <= 0:
        raise OrderPlanError("order notional must be greater than 0")
    requested = _decimal(requested_notional)
    if requested is None or requested <= 0:
        raise OrderPlanError("order notional must be a finite positive number")

    binance_side = "BUY" if signal.side == "LONG" else "SELL"
    stop_price_decimal = _decimal(signal.trigger_price)
    if stop_price_decimal is None or stop_price_decimal <= 0:
        raise OrderPlanError(f"{signal.symbol}: invalid trigger price")

    stop_price = _round_to_step(
        stop_price_decimal,
        rule.price_tick_size,
        rounding=ROUND_UP if binance_side == "BUY" else ROUND_DOWN,
    )
    if stop_price <= 0:
        raise OrderPlanError(f"{signal.symbol}: rounded stop price is invalid")

    limit_price = _entry_limit_price(
        stop_price=stop_price,
        side=signal.side,
        pullback_pct=entry_pullback_pct,
        tick_size=rule.price_tick_size,
    )
    sizing_price = limit_price if entry_mode == "RETEST_LIMIT" else stop_price
    quantity = _round_to_step(requested / sizing_price, rule.quantity_step_size, rounding=ROUND_DOWN)
    if quantity <= 0 or quantity < rule.min_qty:
        raise OrderPlanError(
            f"{signal.symbol}: {requested_notional:g} USDT is below min quantity "
            f"{_format_decimal(rule.min_qty)} at trigger {_format_decimal(stop_price)}"
        )
    if rule.max_qty > 0 and quantity > rule.max_qty:
        quantity = _round_to_step(rule.max_qty, rule.quantity_step_size, rounding=ROUND_DOWN)

    estimated_notional = quantity * sizing_price
    if rule.min_notional > 0 and estimated_notional < rule.min_notional:
        raise OrderPlanError(
            f"{signal.symbol}: rounded order notional {_format_decimal(estimated_notional)} "
            f"is below exchange min notional {_format_decimal(rule.min_notional)}"
        )

    payload = {
        "algoType": "CONDITIONAL",
        "symbol": signal.symbol,
        "side": binance_side,
        "type": "STOP" if entry_mode == "RETEST_LIMIT" else "STOP_MARKET",
        "quantity": _format_decimal(quantity),
        "triggerPrice": _format_decimal(stop_price),
        "workingType": working_type,
        "newOrderRespType": "ACK",
        "clientAlgoId": client_order_id,
    }
    if entry_mode == "RETEST_LIMIT":
        payload["price"] = _format_decimal(limit_price)
        payload["timeInForce"] = "GTC"
    else:
        payload["priceProtect"] = "true" if price_protect else "false"
    if hedge_mode:
        payload["positionSide"] = signal.side

    return ConditionalOrderPlan(
        role="ENTRY",
        symbol=signal.symbol,
        interval=signal.interval,
        signal_side=signal.side,
        binance_side=binance_side,
        order_type="STOP_LIMIT" if entry_mode == "RETEST_LIMIT" else "STOP_MARKET",
        trigger_price=payload["triggerPrice"],
        limit_price=payload.get("price", ""),
        quantity=payload["quantity"],
        requested_notional=_format_decimal(requested),
        estimated_notional=_format_decimal(estimated_notional),
        client_order_id=client_order_id,
        payload=payload,
    )


def build_exit_order_plans(
    signal: BreakoutSignal,
    rule: TradingRule,
    entry_quantity: str,
    stop_client_order_id: str,
    target_client_order_ids: list[str],
    target_splits_pct: list[float],
    trailing_client_order_id: str | None,
    trailing_callback_pct: float | None,
    trailing_quantity_pct: float,
    working_type: str,
    price_protect: bool,
    hedge_mode: bool,
    trail_activation_price: float = 0.0,
) -> list[ConditionalOrderPlan]:
    plans = [
        _build_close_position_plan(
            signal=signal,
            rule=rule,
            role="STOP_LOSS",
            client_order_id=stop_client_order_id,
            working_type=working_type,
            price_protect=price_protect,
            hedge_mode=hedge_mode,
        )
    ]
    if len(target_client_order_ids) != len(target_splits_pct):
        raise OrderPlanError("take-profit client IDs and splits must have the same length")
    entry_quantity_decimal = _decimal(entry_quantity)
    if entry_quantity_decimal is None or entry_quantity_decimal <= 0:
        raise OrderPlanError(f"{signal.symbol}: invalid entry quantity for exit orders")

    # Adapt the scale-out legs so every placed order clears the exchange minimum
    # notional. A small position cannot be split into many legs - each slice would
    # fall under the minimum and be rejected. The stop loss always covers all.
    trailing_on = (
        bool(trailing_client_order_id) and trailing_callback_pct is not None and trailing_quantity_pct > 0
    )
    reference_price = _decimal(signal.trigger_price) or Decimal("0")
    position_notional = entry_quantity_decimal * reference_price
    fitted_splits, fitted_runner_pct = _fit_exit_legs(
        position_notional, rule.min_notional, target_splits_pct, trailing_on, trailing_quantity_pct
    )

    placed_tp_qty = Decimal("0")
    for index, (client_order_id, split_pct) in enumerate(zip(target_client_order_ids, fitted_splits), start=1):
        try:
            tp_plan = _build_partial_take_profit_plan(
                signal=signal,
                rule=rule,
                entry_quantity=entry_quantity_decimal,
                target_index=index,
                target_count=len(fitted_splits),
                split_pct=split_pct,
                client_order_id=client_order_id,
                working_type=working_type,
                price_protect=price_protect,
                hedge_mode=hedge_mode,
            )
        except OrderPlanError:
            continue  # a rounding edge pushed this leg under the minimum - the SL still covers it
        plans.append(tp_plan)
        placed_tp_qty += _decimal(tp_plan.quantity) or Decimal("0")

    if trailing_on and fitted_runner_pct > 0:
        # The trail consumes whatever the TPs didn't take. Computing it as a
        # percentage with independent floor-rounding leaves lot-step dust
        # (e.g. 12.7 split 50/50 with step 0.1 produces TP1=6.3 + trail=6.3,
        # losing 0.1 to dust). Subtracting the actual placed TP quantity makes
        # TP1 + trail equal the entry exactly.
        runner_remainder = entry_quantity_decimal - placed_tp_qty
        try:
            plans.append(
                _build_trailing_stop_plan(
                    signal=signal,
                    rule=rule,
                    entry_quantity=entry_quantity_decimal,
                    quantity_pct=fitted_runner_pct,
                    callback_pct=trailing_callback_pct,
                    client_order_id=trailing_client_order_id,
                    working_type=working_type,
                    hedge_mode=hedge_mode,
                    quantity_override=runner_remainder if runner_remainder > 0 else None,
                    activation_price_override=trail_activation_price if trail_activation_price > 0 else None,
                )
            )
        except OrderPlanError:
            pass  # runner slice under the minimum - skip it, the SL covers the remainder

    return plans


def _fit_exit_legs(
    position_notional: Decimal,
    min_notional: Decimal,
    tp_splits_pct: list[float],
    trailing_on: bool,
    runner_pct: float,
) -> tuple[list[float], float]:
    """Adapt scale-out legs so each placed order clears the exchange minimum notional.

    A small position cannot be split into many legs - each would fall under the
    exchange minimum and get rejected. This reduces the leg count (switching to an
    even split) so every leg is viable. The stop loss covers the full position
    regardless, so dropped legs are never left unprotected.
    """
    runner = runner_pct if (trailing_on and runner_pct > 0) else 0.0
    tp_splits = [float(split) for split in tp_splits_pct]
    if min_notional <= 0 or position_notional <= 0:
        return tp_splits, runner
    desired = len(tp_splits) + (1 if runner > 0 else 0)
    if desired <= 0:
        return tp_splits, runner
    affordable = int(position_notional // min_notional)
    if affordable >= desired:
        return tp_splits, runner  # the requested ladder fits as-is
    if affordable <= 1:
        return [100.0], 0.0  # only one viable leg - a single full take-profit, no runner
    leg = 100.0 / affordable
    if runner > 0:
        return [leg] * (affordable - 1), leg
    return [leg] * affordable, 0.0


def _build_partial_take_profit_plan(
    signal: BreakoutSignal,
    rule: TradingRule,
    entry_quantity: Decimal,
    target_index: int,
    target_count: int,
    split_pct: float,
    client_order_id: str,
    working_type: str,
    price_protect: bool,
    hedge_mode: bool,
) -> ConditionalOrderPlan:
    if split_pct <= 0:
        raise OrderPlanError(f"{signal.symbol}: take-profit split must be positive")
    binance_side, order_type, trigger_price, rounding = _take_profit_params(signal, target_index, target_count)
    quantity = _quantity_from_pct(
        symbol=signal.symbol,
        entry_quantity=entry_quantity,
        quantity_pct=split_pct,
        trigger_price=trigger_price,
        rule=rule,
        role=f"TAKE_PROFIT_{target_index}",
    )
    payload = {
        "algoType": "CONDITIONAL",
        "symbol": signal.symbol,
        "side": binance_side,
        "type": order_type,
        "quantity": _format_decimal(quantity),
        "triggerPrice": _format_decimal(_round_to_step(trigger_price, rule.price_tick_size, rounding=rounding)),
        "workingType": working_type,
        "priceProtect": "true" if price_protect else "false",
        "newOrderRespType": "ACK",
        "clientAlgoId": client_order_id,
    }
    if hedge_mode:
        payload["positionSide"] = signal.side
    else:
        payload["reduceOnly"] = "true"

    return ConditionalOrderPlan(
        role=f"TAKE_PROFIT_{target_index}",
        symbol=signal.symbol,
        interval=signal.interval,
        signal_side=signal.side,
        binance_side=binance_side,
        order_type=order_type,
        trigger_price=payload["triggerPrice"],
        limit_price="",
        quantity=payload["quantity"],
        requested_notional="0",
        estimated_notional=_format_decimal(quantity * _decimal(payload["triggerPrice"])),
        client_order_id=client_order_id,
        payload=payload,
    )


def _build_trailing_stop_plan(
    signal: BreakoutSignal,
    rule: TradingRule,
    entry_quantity: Decimal,
    quantity_pct: float,
    callback_pct: float,
    client_order_id: str,
    working_type: str,
    hedge_mode: bool,
    quantity_override: Decimal | None = None,
    activation_price_override: Decimal | None = None,
) -> ConditionalOrderPlan:
    callback_rate = format_trailing_callback_pct(callback_pct)
    binance_side, activation_price, rounding = _trailing_stop_params(signal)
    if quantity_override is not None and quantity_override > 0:
        # Caller wants the trail to consume an exact remainder (entry - sum(TPs))
        # so TP + trail close the position with no lot-step dust left behind.
        quantity = _round_to_step(quantity_override, rule.quantity_step_size, rounding=ROUND_DOWN)
        if quantity <= 0 or quantity < rule.min_qty:
            raise OrderPlanError(
                f"{signal.symbol}: trailing remainder {_format_decimal(quantity_override)} below min quantity"
            )
        estimated_notional = quantity * activation_price
        if rule.min_notional > 0 and estimated_notional < rule.min_notional:
            raise OrderPlanError(
                f"{signal.symbol}: trailing remainder notional {_format_decimal(estimated_notional)} "
                f"below exchange min notional {_format_decimal(rule.min_notional)}"
            )
    else:
        quantity = _quantity_from_pct(
            symbol=signal.symbol,
            entry_quantity=entry_quantity,
            quantity_pct=quantity_pct,
            trigger_price=activation_price,
            rule=rule,
            role="TRAILING_STOP",
        )
    rounded_activation = _round_to_step(activation_price, rule.price_tick_size, rounding=rounding)
    payload = {
        "algoType": "CONDITIONAL",
        "symbol": signal.symbol,
        "side": binance_side,
        "type": "TRAILING_STOP_MARKET",
        "quantity": _format_decimal(quantity),
        "callbackRate": callback_rate,
        "workingType": working_type,
        "newOrderRespType": "ACK",
        "clientAlgoId": client_order_id,
    }
    # Trail activation gate: if an activation price is provided, Binance will
    # only start tracking the trailing stop once the mark price crosses this
    # level. This prevents the trail from wicking out on the entry-bar pullback
    # and gives the trade room to develop. Omitted when 0 (legacy: trail active
    # from placement, which doubles as an immediate-protection floor).
    if activation_price_override is not None and activation_price_override > 0:
        gate_price = _round_to_step(activation_price_override, rule.price_tick_size, rounding=rounding)
        payload["activationPrice"] = _format_decimal(gate_price)
    if hedge_mode:
        payload["positionSide"] = signal.side
    else:
        payload["reduceOnly"] = "true"

    return ConditionalOrderPlan(
        role="TRAILING_STOP",
        symbol=signal.symbol,
        interval=signal.interval,
        signal_side=signal.side,
        binance_side=binance_side,
        order_type="TRAILING_STOP_MARKET",
        trigger_price=_format_decimal(rounded_activation),
        limit_price="",
        quantity=payload["quantity"],
        requested_notional="0",
        estimated_notional=_format_decimal(quantity * rounded_activation),
        client_order_id=client_order_id,
        payload=payload,
    )


def _build_close_position_plan(
    signal: BreakoutSignal,
    rule: TradingRule,
    role: str,
    client_order_id: str,
    working_type: str,
    price_protect: bool,
    hedge_mode: bool,
) -> ConditionalOrderPlan:
    binance_side, order_type, raw_trigger, rounding = _exit_order_params(signal, role)
    trigger_decimal = _decimal(raw_trigger)
    if trigger_decimal is None or trigger_decimal <= 0:
        raise OrderPlanError(f"{signal.symbol}: invalid {role.lower()} trigger price")
    trigger_price = _round_to_step(trigger_decimal, rule.price_tick_size, rounding=rounding)
    if trigger_price <= 0:
        raise OrderPlanError(f"{signal.symbol}: rounded {role.lower()} trigger price is invalid")

    payload = {
        "algoType": "CONDITIONAL",
        "symbol": signal.symbol,
        "side": binance_side,
        "type": order_type,
        "triggerPrice": _format_decimal(trigger_price),
        "workingType": working_type,
        "priceProtect": "true" if price_protect else "false",
        "closePosition": "true",
        "newOrderRespType": "ACK",
        "clientAlgoId": client_order_id,
    }
    if hedge_mode:
        payload["positionSide"] = signal.side

    return ConditionalOrderPlan(
        role=role,
        symbol=signal.symbol,
        interval=signal.interval,
        signal_side=signal.side,
        binance_side=binance_side,
        order_type=order_type,
        trigger_price=payload["triggerPrice"],
        limit_price="",
        quantity="close-all",
        requested_notional="0",
        estimated_notional="close-position",
        client_order_id=client_order_id,
        payload=payload,
    )


def _take_profit_params(signal: BreakoutSignal, target_index: int, target_count: int) -> tuple[str, str, Decimal, str]:
    if target_count <= 0:
        raise OrderPlanError("take-profit count must be positive")
    entry = _decimal(signal.trigger_price)
    target = _decimal(signal.target_price)
    if entry is None or target is None:
        raise OrderPlanError(f"{signal.symbol}: invalid take-profit target")
    fraction = Decimal(target_index) / Decimal(target_count)
    if signal.side == "LONG":
        trigger = entry + (target - entry) * fraction
        return "SELL", "TAKE_PROFIT_MARKET", trigger, ROUND_UP
    trigger = entry - (entry - target) * fraction
    return "BUY", "TAKE_PROFIT_MARKET", trigger, ROUND_DOWN


def _trailing_stop_params(signal: BreakoutSignal) -> tuple[str, Decimal, str]:
    activation_price = _decimal(signal.trigger_price)
    if activation_price is None or activation_price <= 0:
        raise OrderPlanError(f"{signal.symbol}: invalid trailing activation price")
    if signal.side == "LONG":
        return "SELL", activation_price, ROUND_UP
    return "BUY", activation_price, ROUND_DOWN


def _entry_limit_price(stop_price: Decimal, side: str, pullback_pct: float, tick_size: Decimal) -> Decimal:
    pullback = _decimal(max(pullback_pct, 0.0))
    if pullback is None or pullback <= 0:
        return stop_price
    fraction = pullback / Decimal("100")
    if side == "LONG":
        return _round_to_step(stop_price * (Decimal("1") - fraction), tick_size, rounding=ROUND_DOWN)
    return _round_to_step(stop_price * (Decimal("1") + fraction), tick_size, rounding=ROUND_UP)


def _quantity_from_pct(
    symbol: str,
    entry_quantity: Decimal,
    quantity_pct: float,
    trigger_price: Decimal,
    rule: TradingRule,
    role: str,
) -> Decimal:
    quantity_pct_decimal = _decimal(quantity_pct)
    if quantity_pct_decimal is None or quantity_pct_decimal <= 0:
        raise OrderPlanError(f"{symbol}: {role} quantity percent must be positive")
    quantity = _round_to_step(entry_quantity * quantity_pct_decimal / Decimal("100"), rule.quantity_step_size, rounding=ROUND_DOWN)
    if quantity <= 0 or quantity < rule.min_qty:
        raise OrderPlanError(f"{symbol}: {role} split is below min quantity {_format_decimal(rule.min_qty)}")
    estimated_notional = quantity * trigger_price
    if rule.min_notional > 0 and estimated_notional < rule.min_notional:
        raise OrderPlanError(
            f"{symbol}: {role} notional {_format_decimal(estimated_notional)} "
            f"is below exchange min notional {_format_decimal(rule.min_notional)}"
        )
    return quantity


def format_trailing_callback_pct(callback_pct: Any) -> str:
    """Return a Binance-safe trailing callback percentage string.

    Binance Futures rejects callbacks outside 0.1..10 and is picky about
    excess decimal precision. Use one-decimal step formatting for both newly
    built plans and old pending plans before retrying placement.
    """
    value = _decimal(callback_pct)
    if value is None:
        raise OrderPlanError("trailing callback must be a finite decimal")
    value = max(TRAILING_CALLBACK_MIN_PCT, min(TRAILING_CALLBACK_MAX_PCT, value))
    value = (value / TRAILING_CALLBACK_STEP_PCT).to_integral_value(rounding=ROUND_DOWN) * TRAILING_CALLBACK_STEP_PCT
    return _format_decimal(value)


def _exit_order_params(signal: BreakoutSignal, role: str) -> tuple[str, str, float, str]:
    if role == "STOP_LOSS":
        if signal.side == "LONG":
            return "SELL", "STOP_MARKET", signal.stop_price, ROUND_DOWN
        return "BUY", "STOP_MARKET", signal.stop_price, ROUND_UP
    if role == "TAKE_PROFIT":
        if signal.side == "LONG":
            return "SELL", "TAKE_PROFIT_MARKET", signal.target_price, ROUND_UP
        return "BUY", "TAKE_PROFIT_MARKET", signal.target_price, ROUND_DOWN
    raise OrderPlanError(f"unsupported exit order role: {role}")


def _round_to_step(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=rounding) * step


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _decimal(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
    except Exception:
        return None
    if not result.is_finite():
        return None
    return result
