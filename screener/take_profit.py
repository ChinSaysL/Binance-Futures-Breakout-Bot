from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .breakout import BreakoutSignal


@dataclass(frozen=True)
class TakeProfitProfile:
    signal: BreakoutSignal
    tp_splits_pct: list[float]
    runner_pct: float
    conviction: float
    target_multiplier: float


def smart_take_profit_profile(
    signal: BreakoutSignal,
    tp_count: int,
    trailing_stop: bool,
    base_runner_pct: float,
    max_target_multiplier: float = 2.5,
    min_runner_pct: float = 20.0,
    max_runner_pct: float = 55.0,
) -> TakeProfitProfile:
    """Build an adaptive TP ladder from signal quality.

    High-conviction signals get a farther final target, a larger trailing runner,
    and less quantity sold at the earliest TP levels.
    """
    tp_count = max(int(tp_count), 1)
    conviction = _conviction_score(signal)
    target_multiplier = 1.0 + (max(max_target_multiplier, 1.0) - 1.0) * conviction
    expanded_signal = _retarget_signal(signal, target_multiplier)

    if trailing_stop:
        runner_pct = min_runner_pct + (max_runner_pct - min_runner_pct) * conviction
        runner_pct = _clamp(runner_pct, 0.0, 95.0)
        if base_runner_pct > 0:
            runner_pct = max(runner_pct, min(base_runner_pct, 95.0))
    else:
        runner_pct = 0.0

    fixed_tp_pct = max(100.0 - runner_pct, 0.0)
    return TakeProfitProfile(
        signal=expanded_signal,
        tp_splits_pct=_ladder_splits(tp_count, fixed_tp_pct, conviction),
        runner_pct=runner_pct,
        conviction=conviction,
        target_multiplier=target_multiplier,
    )


def equal_take_profit_profile(
    signal: BreakoutSignal,
    tp_count: int,
    trailing_stop: bool,
    runner_pct: float,
) -> TakeProfitProfile:
    tp_count = max(int(tp_count), 1)
    runner = runner_pct if trailing_stop else 0.0
    fixed_tp_pct = max(100.0 - runner, 0.0)
    split = fixed_tp_pct / tp_count
    return TakeProfitProfile(
        signal=signal,
        tp_splits_pct=[split for _ in range(tp_count)],
        runner_pct=runner,
        conviction=0.0,
        target_multiplier=1.0,
    )


def _conviction_score(signal: BreakoutSignal) -> float:
    volume_score = _clamp((signal.volume_ratio - 1.3) / 4.0, 0.0, 1.0)
    atr_score = _clamp((signal.atr_pct - 0.010) / 0.060, 0.0, 1.0)
    trend_score = _clamp(signal.trend_score, 0.0, 1.0)
    range_score = _clamp(signal.range_pct_24h / 45.0, 0.0, 1.0)
    rr_score = _clamp((signal.reward_risk - 1.2) / 2.8, 0.0, 1.0)

    close_score = _clamp((signal.close_position - 0.55) / 0.35, 0.0, 1.0)
    status_bonus = 0.08 if signal.status in {"BREAKOUT", "BREAKDOWN"} else 0.0

    score = (
        volume_score * 0.24
        + atr_score * 0.22
        + trend_score * 0.18
        + range_score * 0.14
        + close_score * 0.12
        + rr_score * 0.10
        + status_bonus
    )
    return _clamp(score, 0.0, 1.0)


def _retarget_signal(signal: BreakoutSignal, target_multiplier: float) -> BreakoutSignal:
    trigger = signal.trigger_price
    target = signal.target_price
    if trigger <= 0 or target <= 0:
        return signal

    if signal.side == "SHORT":
        distance = max(trigger - target, 0.0)
        new_target = max(trigger - distance * target_multiplier, 1e-12)
        reward_pct = max(trigger / max(new_target, 1e-12) - 1.0, 0.0)
    else:
        distance = max(target - trigger, 0.0)
        new_target = trigger + distance * target_multiplier
        reward_pct = max(new_target / max(trigger, 1e-12) - 1.0, 0.0)

    reward_risk = reward_pct / signal.risk_pct if signal.risk_pct > 0 else signal.reward_risk
    return replace(
        signal,
        target_price=new_target,
        reward_pct=reward_pct,
        reward_risk=reward_risk,
    )


def _ladder_splits(tp_count: int, fixed_tp_pct: float, conviction: float) -> list[float]:
    if fixed_tp_pct <= 0:
        return []
    if tp_count <= 1:
        return [fixed_tp_pct]

    tilt = (conviction - 0.5) * 0.9
    weights: list[float] = []
    for index in range(tp_count):
        centered = (index / (tp_count - 1)) * 2.0 - 1.0
        weights.append(math.exp(tilt * centered))
    total = sum(weights)
    return [fixed_tp_pct * weight / total for weight in weights]


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
