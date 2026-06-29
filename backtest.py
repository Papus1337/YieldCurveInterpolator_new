"""
yield_curve.backtest
====================

Модуль бэктеста торговых стратегий на сигналах
модели кривой депозитных ставок.

Философия
---------
Бэктест симулирует реальную торговлю:
- Сигналы генерируются на конец дня t.
- Позиции открываются/ребалансируются на день t+1.
- PnL считается с учётом duration и транзакционных издержек.
- Риск-метрики (Sharpe, Sortino, MaxDD) рассчитываются
  по дневным данным с годовой аннуализацией.

Поддерживаемые стратегии
------------------------
'directional'
    Торговля в направлении движения ставки.
    Если модель предсказывает рост — занимаем short,
    если падение — long.

'mean_reversion'
    Торговля на возврат к модельной кривой.
    Если рыночная ставка выше модельной (завышена) —
    открываем long, ожидая падения.

'carry'
    Фиксация положительного спреда между длинным
    и коротким концом (roll-down эффект).

Структура
---------
CurveBacktester   — основной класс для запуска бэктеста
BacktestResult    — контейнер с результатами и метриками
plot_*            — функции визуализации (Plotly)

Автор:
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


###############################################################################
# Constants
###############################################################################

# Финансовые параметры по умолчанию
DEFAULT_TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.0           # годовая безрисковая ставка (в долях)
DEFAULT_TRANSACTION_COST_BPS = 5.0     # cost в базисных пунктах на ребалансировку
DEFAULT_SLIPPAGE_BPS = 2.0             # slippage в бп
DEFAULT_NOTIONAL = 1_000_000.0         # нотионал на бакет
DEFAULT_BPS = 100.0                    # 1% = 100 бп

# Duration по бакетам (в годах) — аппроксимация Macaulay duration
DEFAULT_DURATION_BY_BUCKET = {
    1: 1 / 365,
    7: 7 / 365,
    14: 14 / 365,
    31: 31 / 365,
    61: 61 / 365,
    91: 91 / 365,
    181: 181 / 365,
}

# Визуальный стиль (совместимо с plotting.py)
COLOR_EQUITY = "#1f77b4"
COLOR_DRAWDOWN = "#d62728"
COLOR_BENCHMARK = "#7f7f7f"
COLOR_POSITIVE = "#2ca02c"
COLOR_NEGATIVE = "#d62728"
DEFAULT_TEMPLATE = "plotly_white"
DEFAULT_FONT_FAMILY = "Segoe UI, Arial, sans-serif"
DEFAULT_FONT_SIZE = 12


###############################################################################
# Signal generation
###############################################################################

def generate_directional_signals(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
) -> pd.DataFrame:
    """
    Генерация сигналов направленной стратегии.

    Логика:
    - Модель предсказывает ставку НИЖЕ рыночной (predicted < actual)
      → рынок должен упасть → открываем LONG (+1).
    - Модель предсказывает ставку ВЫШЕ рыночной (predicted > actual)
      → рынок должен вырасти → открываем SHORT (-1).
    - Сигнал = sign(actual - predicted).

    Returns
    -------
    DataFrame
        Матрица сигналов той же формы, что и actual.
        Значения: -1, 0, +1.
    """

    common_idx = predicted.index.intersection(actual.index)
    common_cols = predicted.columns.intersection(actual.columns)

    p = predicted.loc[common_idx, common_cols]
    a = actual.loc[common_idx, common_cols]

    diff = a - p
    signals = np.sign(diff)

    # Обработка NaN: если нет данных — позиция 0
    signals = signals.fillna(0.0)

    return signals


def generate_mean_reversion_signals(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    zscore_threshold: float = 0.5,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Генерация сигналов mean-reversion стратегии.

    Сигнал генерируется, только если спред (actual - predicted)
    превышает zscore_threshold стандартных отклонений от своего
    скользящего среднего. Это фильтрует шум.

    Parameters
    ----------
    predicted, actual
    zscore_threshold
        Порог отсечки в стандартных отклонениях.
    lookback
        Окно для расчёта скользящего среднего и std.
    """

    common_idx = predicted.index.intersection(actual.index)
    common_cols = predicted.columns.intersection(actual.columns)

    p = predicted.loc[common_idx, common_cols]
    a = actual.loc[common_idx, common_cols]

    spread = a - p

    roll_mean = spread.rolling(window=lookback, min_periods=1).mean()
    roll_std = spread.rolling(window=lookback, min_periods=1).std()
    roll_std = roll_std.replace(0, np.nan)

    zscore = (spread - roll_mean) / roll_std
    zscore = zscore.fillna(0.0)

    # Сигнал: сильный положительный спред → LONG (ждём падения)
    #         сильный отрицательный спред → SHORT (ждём роста)
    signals = np.sign(zscore)
    signals = signals.where(zscore.abs() > zscore_threshold, 0.0)

    return signals


def generate_carry_signals(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    long_bucket: int = 181,
    short_bucket: int = 7,
) -> pd.DataFrame:
    """
    Генерация сигналов carry-стратегии (roll-down).

    Занимаем long на длинном конце (фиксируем высокую ставку),
    short на коротком (платим низкую ставку).
    Зарабатываем на положительном наклоне кривой.

    Parameters
    ----------
    predicted, actual
    long_bucket
        Бакет для long-позиции.
    short_bucket
        Бакет для short-позиции.
    """

    signals = pd.DataFrame(0.0, index=actual.index, columns=actual.columns)

    if long_bucket in signals.columns:
        signals[long_bucket] = 1.0
    if short_bucket in signals.columns:
        signals[short_bucket] = -1.0

    return signals


SIGNAL_GENERATORS = {
    "directional": generate_directional_signals,
    "mean_reversion": generate_mean_reversion_signals,
    "carry": generate_carry_signals,
}


###############################################################################
# BacktestResult — контейнер результатов
###############################################################################

@dataclass
class BacktestResult:
    """
    Контейнер с результатами бэктеста.

    Attributes
    ----------
    strategy_name : str
    equity_curve : pd.Series
    daily_pnl : pd.DataFrame
    positions : pd.DataFrame
    trades : pd.DataFrame
    signals : pd.DataFrame
    predicted : pd.DataFrame
    actual : pd.DataFrame
    params : dict
    """

    strategy_name: str
    equity_curve: pd.Series
    daily_pnl: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    predicted: pd.DataFrame
    actual: pd.DataFrame
    params: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        """Общая доходность (в денежных единицах)."""
        return float(self.equity_curve.iloc[-1] - self.equity_curve.iloc[0])

    def total_return_pct(self) -> float:
        """Общая доходность в процентах от начального капитала."""
        initial = self.equity_curve.iloc[0]
        if initial == 0:
            return float("nan")
        return float((self.equity_curve.iloc[-1] - initial) / initial * 100)

    def annualized_return(self) -> float:
        """Годовая доходность (CAGR)."""
        n_days = len(self.equity_curve)
        if n_days < 2:
            return float("nan")

        total_factor = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]
        if total_factor <= 0:
            return float("nan")

        years = (n_days - 1) / DEFAULT_TRADING_DAYS_PER_YEAR
        return float(total_factor ** (1 / years) - 1)

    def annualized_volatility(self) -> float:
        """Годовая волатильность дневных PnL."""
        daily_returns = self.equity_curve.diff().dropna()
        if daily_returns.empty:
            return float("nan")

        # Нормируем на начальный капитал
        initial = self.equity_curve.iloc[0]
        if initial == 0:
            return float("nan")

        daily_returns_pct = daily_returns / initial
        return float(daily_returns_pct.std() * np.sqrt(DEFAULT_TRADING_DAYS_PER_YEAR))

    def sharpe_ratio(
        self,
        risk_free: float = DEFAULT_RISK_FREE_RATE,
    ) -> float:
        """
        Коэффициент Шарпа (годовой).

        Sharpe = (R - Rf) / sigma
        """

        ann_ret = self.annualized_return()
        ann_vol = self.annualized_volatility()

        if ann_vol == 0 or np.isnan(ann_vol):
            return float("nan")

        return float((ann_ret - risk_free) / ann_vol)

    def sortino_ratio(
        self,
        risk_free: float = DEFAULT_RISK_FREE_RATE,
    ) -> float:
        """
        Коэффициент Сортино.

        Как Шарпа, но в знаменателе — только downside deviation
        (волатильность отрицательных дневных доходностей).
        """

        daily_returns = self.equity_curve.diff().dropna()
        initial = self.equity_curve.iloc[0]
        if initial == 0:
            return float("nan")

        daily_returns_pct = daily_returns / initial
        negative = daily_returns_pct[daily_returns_pct < 0]

        if negative.empty:
            return float("inf")

        downside_std = float(negative.std() * np.sqrt(DEFAULT_TRADING_DAYS_PER_YEAR))
        if downside_std == 0:
            return float("inf")

        ann_ret = self.annualized_return()
        return float((ann_ret - risk_free) / downside_std)

    def max_drawdown(self) -> float:
        """
        Максимальная просадка (в долях от пика).

        Возвращает отрицательное число, например -0.15 = 15% просадка.
        """

        equity = self.equity_curve.values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak

        return float(np.min(drawdown))

    def max_drawdown_duration(self) -> int:
        """
        Максимальная длительность просадки (в торговых днях).

        Период от последнего пика до нового максимума.
        """

        equity = self.equity_curve.values
        peak = np.maximum.accumulate(equity)
        in_drawdown = equity < peak

        max_duration = 0
        current_duration = 0

        for dd in in_drawdown:
            if dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def calmar_ratio(self) -> float:
        """
        Коэффициент Кальмара.

        Calmar = annualized_return / abs(max_drawdown)
        """

        ann_ret = self.annualized_return()
        mdd = self.max_drawdown()

        if mdd == 0 or np.isnan(mdd):
            return float("nan")

        return float(ann_ret / abs(mdd))

    def win_rate(self) -> float:
        """Доля прибыльных дней."""

        daily_pnl = self.equity_curve.diff().dropna()
        if daily_pnl.empty:
            return float("nan")

        return float((daily_pnl > 0).mean())

    def profit_factor(self) -> float:
        """
        Profit factor.

        Отношение суммарной прибыли к суммарному убытку.
        """

        daily_pnl = self.equity_curve.diff().dropna()
        gains = daily_pnl[daily_pnl > 0].sum()
        losses = -daily_pnl[daily_pnl < 0].sum()

        if losses == 0:
            return float("inf") if gains > 0 else float("nan")

        return float(gains / losses)

    def n_trades(self) -> int:
        """Количество сделок (ребалансировок позиций)."""
        return int(len(self.trades))

    def turnover(self) -> float:
        """
        Средний дневной оборот (в денежных единицах).

        Sum of |delta_position| / n_days.
        """

        if self.trades.empty:
            return 0.0

        daily_turnover = self.trades.groupby("date")["abs_position_change"].sum()
        n_days = len(self.positions)

        return float(daily_turnover.sum() / n_days)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Сводный отчёт по результатам бэктеста.
        """

        return {
            "strategy": self.strategy_name,
            "n_days": len(self.equity_curve),
            "start_date": self.equity_curve.index[0],
            "end_date": self.equity_curve.index[-1],
            "total_return": self.total_return(),
            "total_return_pct": self.total_return_pct(),
            "annualized_return": self.annualized_return(),
            "annualized_volatility": self.annualized_volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_duration": self.max_drawdown_duration(),
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "n_trades": self.n_trades(),
            "turnover": self.turnover(),
            "params": self.params,
        }

    def print_summary(self) -> None:
        """Печать сводного отчёта."""

        s = self.summary()

        print("=" * 70)
        print(f"BACKTEST REPORT — {s['strategy'].upper()}")
        print("=" * 70)

        print(f"\n[Period]")
        print(f"    Start:              {s['start_date']}")
        print(f"    End:                {s['end_date']}")
        print(f"    Trading days:       {s['n_days']}")

        print(f"\n[Returns]")
        print(f"    Total return:       {s['total_return']:>15,.2f}")
        print(f"    Total return (%):   {s['total_return_pct']:>15.4f}%")
        print(f"    Annualized return:  {s['annualized_return']:>15.4f} "
              f"({s['annualized_return'] * 100:.2f}%)")
        print(f"    Annualized vol:     {s['annualized_volatility']:>15.4f} "
              f"({s['annualized_volatility'] * 100:.2f}%)")

        print(f"\n[Risk-adjusted]")
        print(f"    Sharpe ratio:       {s['sharpe_ratio']:>15.4f}")
        print(f"    Sortino ratio:      {s['sortino_ratio']:>15.4f}")
        print(f"    Calmar ratio:       {s['calmar_ratio']:>15.4f}")
        print(f"    Max drawdown:       {s['max_drawdown']:>15.4f} "
              f"({s['max_drawdown'] * 100:.2f}%)")
        print(f"    Max DD duration:    {s['max_drawdown_duration']:>15d} days")

        print(f"\n[Trading]")
        print(f"    Win rate:           {s['win_rate']:>15.4f} "
              f"({s['win_rate'] * 100:.2f}%)")
        print(f"    Profit factor:      {s['profit_factor']:>15.4f}")
        print(f"    Number of trades:   {s['n_trades']:>15d}")
        print(f"    Avg daily turnover: {s['turnover']:>15,.2f}")

        if s["params"]:
            print(f"\n[Parameters]")
            for k, v in s["params"].items():
                print(f"    {k:22s}: {v}")

        print("\n" + "=" * 70)


###############################################################################
# CurveBacktester — основной класс
###############################################################################

class CurveBacktester:
    """
    Движок бэктеста стратегий на сигналах модели кривой ставок.

    Parameters
    ----------
    strategy : str or callable
        Имя предустановленной стратегии ('directional',
        'mean_reversion', 'carry') или пользовательская функция
        generate_signals(predicted, actual) -> pd.DataFrame.
    strategy_params : dict
        Параметры для генератора сигналов.
    notional : float
        Нотионал на бакет.
    duration_map : dict
        Словарь {bucket: duration_in_years}.
    transaction_cost_bps : float
        Транзакционные издержки в базисных пунктах.
    slippage_bps : float
        Проскальзывание в базисных пунктах.
    risk_free_rate : float
        Годовая безрисковая ставка (в долях).
    signal_lag : int
        Задержка исполнения сигнала в днях.
        signal_lag=1: сигнал на день t → позиция на день t+1.
    rebalance_frequency : int
        Частота ребалансировки в днях (1 = каждый день).
    """

    def __init__(
        self,
        strategy: Union[str, Callable] = "mean_reversion",
        strategy_params: Optional[Dict[str, Any]] = None,
        notional: float = DEFAULT_NOTIONAL,
        duration_map: Optional[Dict[int, float]] = None,
        transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        signal_lag: int = 1,
        rebalance_frequency: int = 1,
    ):
        if callable(strategy):
            self._signal_fn = strategy
            self.strategy_name = getattr(strategy, "__name__", "custom")
        elif isinstance(strategy, str):
            if strategy not in SIGNAL_GENERATORS:
                raise ValueError(
                    f"Unknown strategy: {strategy}. "
                    f"Available: {list(SIGNAL_GENERATORS.keys())}"
                )
            self._signal_fn = SIGNAL_GENERATORS[strategy]
            self.strategy_name = strategy
        else:
            raise TypeError(
                "strategy must be str or callable, "
                f"got {type(strategy).__name__}"
            )

        self.strategy_params = strategy_params or {}
        self.notional = notional
        self.duration_map = duration_map or DEFAULT_DURATION_BY_BUCKET
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.risk_free_rate = risk_free_rate
        self.signal_lag = signal_lag
        self.rebalance_frequency = rebalance_frequency

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def run(
        self,
        predicted: pd.DataFrame,
        actual: pd.DataFrame,
    ) -> BacktestResult:
        """
        Запуск бэктеста.

        Parameters
        ----------
        predicted
            Реконструированная моделью кривая (wide).
        actual
            Реальная рыночная кривая (wide).

        Returns
        -------
        BacktestResult
        """

        # 1. Генерация сигналов
        signals = self._signal_fn(predicted, actual, **self.strategy_params)

        # Приведение к общему индексу
        common_idx = predicted.index.intersection(actual.index).intersection(signals.index)
        common_cols = predicted.columns.intersection(actual.columns).intersection(signals.columns)

        p = predicted.loc[common_idx, common_cols]
        a = actual.loc[common_idx, common_cols]
        s = signals.loc[common_idx, common_cols].fillna(0.0)

        if len(common_idx) < self.signal_lag + 2:
            raise ValueError(
                f"Слишком мало данных: {len(common_idx)} дней, "
                f"требуется минимум {self.signal_lag + 2}."
            )

        # 2. Позиции с учётом signal_lag и rebalance_frequency
        #    Позиция на день t определяется сигналом на день t - signal_lag
        positions = pd.DataFrame(0.0, index=common_idx, columns=common_cols)

        for i, date in enumerate(common_idx):
            if i < self.signal_lag:
                continue
            # Ребалансировка: если i % rebalance_frequency == 0 или первая позиция
            if i % self.rebalance_frequency == 0 or i == self.signal_lag:
                signal_date = common_idx[i - self.signal_lag]
                positions.loc[date] = s.loc[signal_date]
            else:
                positions.loc[date] = positions.iloc[i - 1]

        # 3. Дневной PnL
        #    PnL[t] = sum over buckets:
        #        position[t-1] * (-duration) * (rate[t] - rate[t-1]) * notional / 100
        #
        #    Если position = +1 (long) и ставка падает (rate[t] < rate[t-1])
        #    → PnL > 0 (заработали на росте цены облигации).
        rate_changes = a.diff()  # rate[t] - rate[t-1]
        rate_changes = rate_changes.fillna(0.0)

        # Применяем position[t-1] — позиция, действовавшая в течение дня t
        positions_prev = positions.shift(1).fillna(0.0)

        # Матрица duration по бакетам
        durations = pd.Series(
            [self.duration_map.get(int(b), b / 365.0) for b in common_cols],
            index=common_cols,
        )

        # PnL по каждой ячейке
        pnl_matrix = (
            -positions_prev
            * rate_changes
            * durations
            * (self.notional / DEFAULT_BPS)
        )

        # 4. Транзакционные издержки
        #    cost = |position[t] - position[t-1]| * cost_bps * notional / 10000
        position_changes = positions.diff().fillna(0.0)
        cost_per_unit = (
            self.transaction_cost_bps + self.slippage_bps
        ) * self.notional / 10_000

        tx_costs = position_changes.abs().sum(axis=1) * cost_per_unit

        # 5. Funding cost (стоимость финансирования позиции)
        #    Упрощённо: position * notional * Rf / 252 / 100
        funding = (
            positions_prev.abs().sum(axis=1)
            * self.notional
            * self.risk_free_rate
            / DEFAULT_TRADING_DAYS_PER_YEAR
        )

        # 6. Сводный daily PnL
        daily_pnl_gross = pnl_matrix.sum(axis=1)
        daily_pnl_net = daily_pnl_gross - tx_costs - funding

        # 7. Equity curve
        #    Начинаем с notional * n_buckets, чтобы было от чего считать %
        initial_capital = self.notional * len(common_cols)
        equity = initial_capital + daily_pnl_net.cumsum()
        equity.name = "equity"

        # 8. Trades — детализация ребалансировок
        trades = self._build_trades(position_changes, cost_per_unit)

        # 9. Параметры для отчёта
        params = {
            "notional": self.notional,
            "transaction_cost_bps": self.transaction_cost_bps,
            "slippage_bps": self.slippage_bps,
            "risk_free_rate": self.risk_free_rate,
            "signal_lag": self.signal_lag,
            "rebalance_frequency": self.rebalance_frequency,
            "n_buckets": len(common_cols),
            "initial_capital": initial_capital,
            **self.strategy_params,
        }

        return BacktestResult(
            strategy_name=self.strategy_name,
            equity_curve=equity,
            daily_pnl=pd.DataFrame({
                "gross": daily_pnl_gross,
                "net": daily_pnl_net,
                "tx_costs": tx_costs,
                "funding": funding,
            }),
            positions=positions,
            trades=trades,
            signals=s,
            predicted=p,
            actual=a,
            params=params,
        )

    def _build_trades(
        self,
        position_changes: pd.DataFrame,
        cost_per_unit: float,
    ) -> pd.DataFrame:
        """
        Построение таблицы сделок из матрицы изменений позиций.
        """

        records: List[Dict[str, Any]] = []

        for date in position_changes.index:
            row = position_changes.loc[date]
            for bucket, change in row.items():
                if abs(change) < 1e-9:
                    continue

                records.append({
                    "date": date,
                    "bucket": bucket,
                    "position_change": float(change),
                    "abs_position_change": float(abs(change) * self.notional),
                    "cost": float(abs(change) * cost_per_unit),
                    "direction": "LONG" if change > 0 else "SHORT",
                })

        if not records:
            return pd.DataFrame(
                columns=[
                    "date", "bucket", "position_change",
                    "abs_position_change", "cost", "direction",
                ]
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def run_from_interpolator(
        self,
        interpolator: Any,
        actual: pd.DataFrame,
    ) -> BacktestResult:
        """
        Удобная обёртка: бэктест обученного интерполятора против эталона.
        """

        if not getattr(interpolator, "is_fitted_", False):
            raise RuntimeError("Интерполятор не обучен.")

        return self.run(
            predicted=interpolator.curve_,
            actual=actual,
        )


###############################################################################
# Multi-strategy comparison
###############################################################################

def compare_strategies(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    strategies: Optional[List[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Сравнение нескольких стратегий в табличном виде.

    Parameters
    ----------
    predicted, actual
    strategies
        Список стратегий для сравнения.
        По умолчанию — все предустановленные.
    **kwargs
        Параметры для CurveBacktester (notional, transaction_cost_bps и т.д.).

    Returns
    -------
    DataFrame
        Индекс — стратегии, колонки — метрики.
    """

    if strategies is None:
        strategies = list(SIGNAL_GENERATORS.keys())

    records = []
    for name in strategies:
        bt = CurveBacktester(strategy=name, **kwargs)
        try:
            result = bt.run(predicted, actual)
            summary = result.summary()
            records.append({
                "strategy": name,
                **{k: summary[k] for k in [
                    "total_return_pct",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe_ratio",
                    "sortino_ratio",
                    "calmar_ratio",
                    "max_drawdown",
                    "win_rate",
                    "profit_factor",
                    "n_trades",
                ]},
            })
        except Exception as e:
            records.append({
                "strategy": name,
                "error": str(e),
            })

    return pd.DataFrame(records).set_index("strategy")


###############################################################################
# Visualization
###############################################################################

def plot_equity_curve(
    result: BacktestResult,
    benchmark: Optional[pd.Series] = None,
    title: Optional[str] = None,
    width: int = 1100,
    height: int = 550,
) -> go.Figure:
    """
    График equity curve с опциональным бенчмарком.

    Parameters
    ----------
    result
        Результат бэктеста.
    benchmark
        Эталонная equity curve для сравнения.
    title
    width, height
    """

    if title is None:
        title = f"Equity Curve — стратегия {result.strategy_name}"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index.tolist(),
        y=result.equity_curve.values.tolist(),
        mode="lines",
        name=result.strategy_name,
        line=dict(color=COLOR_EQUITY, width=2),
    ))

    if benchmark is not None:
        common_idx = result.equity_curve.index.intersection(benchmark.index)
        fig.add_trace(go.Scatter(
            x=common_idx.tolist(),
            y=benchmark.loc[common_idx].values.tolist(),
            mode="lines",
            name="Benchmark",
            line=dict(color=COLOR_BENCHMARK, width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family=DEFAULT_FONT_FAMILY),
            x=0.5,
        ),
        template=DEFAULT_TEMPLATE,
        font=dict(family=DEFAULT_FONT_FAMILY, size=DEFAULT_FONT_SIZE),
        width=width,
        height=height,
        xaxis_title="Дата",
        yaxis_title="Equity, ден. ед.",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified",
    )

    return fig


def plot_drawdown(
    result: BacktestResult,
    title: Optional[str] = None,
    width: int = 1100,
    height: int = 350,
) -> go.Figure:
    """
    График просадки (drawdown) от пика.
    """

    if title is None:
        title = f"Drawdown — стратегия {result.strategy_name}"

    equity = result.equity_curve.values
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100  # в процентах

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index.tolist(),
        y=drawdown.tolist(),
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color=COLOR_DRAWDOWN, width=1),
        fillcolor="rgba(214, 39, 40, 0.3)",
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family=DEFAULT_FONT_FAMILY),
            x=0.5,
        ),
        template=DEFAULT_TEMPLATE,
        font=dict(family=DEFAULT_FONT_FAMILY, size=DEFAULT_FONT_SIZE),
        width=width,
        height=height,
        xaxis_title="Дата",
        yaxis_title="Drawdown, %",
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified",
    )

    return fig


def plot_daily_pnl(
    result: BacktestResult,
    title: Optional[str] = None,
    width: int = 1100,
    height: int = 400,
) -> go.Figure:
    """
    График дневного PnL (bar chart).
    """

    if title is None:
        title = f"Daily PnL — стратегия {result.strategy_name}"

    pnl = result.daily_pnl["net"]
    colors = [
        COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE
        for v in pnl.values
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pnl.index.tolist(),
        y=pnl.values.tolist(),
        marker_color=colors,
        name="Daily PnL",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family=DEFAULT_FONT_FAMILY),
            x=0.5,
        ),
        template=DEFAULT_TEMPLATE,
        font=dict(family=DEFAULT_FONT_FAMILY, size=DEFAULT_FONT_SIZE),
        width=width,
        height=height,
        xaxis_title="Дата",
        yaxis_title="PnL, ден. ед.",
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified",
    )

    return fig


def plot_monthly_returns(
    result: BacktestResult,
    title: Optional[str] = None,
    width: int = 1100,
    height: int = 500,
) -> go.Figure:
    """
    Heatmap месячных доходностей.
    """

    if title is None:
        title = f"Месячные доходности — стратегия {result.strategy_name}"

    equity = result.equity_curve
    monthly = equity.resample("M").last().pct_change().dropna() * 100

    # Группировка по годам и месяцам
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return_pct": monthly.values,
    })

    pivot = monthly_df.pivot(index="year", columns="month", values="return_pct")
    pivot = pivot.reindex(columns=range(1, 13))

    ru_months = [
        "Янв", "Фев", "Мар", "Апр", "Май", "Июн",
        "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек",
    ]

    max_abs = float(np.nanmax(np.abs(pivot.values)))

    fig = go.Figure(data=go.Heatmap(
        x=ru_months,
        y=pivot.index.tolist(),
        z=pivot.values,
        colorscale="RdYlGn",
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        text=np.round(pivot.values, 2),
        texttemplate="%{text}%",
        textfont=dict(size=10),
        colorbar=dict(title="%", len=0.7),
        hovertemplate=(
            "Год: %{y}<br>"
            "Месяц: %{x}<br>"
            "Доходность: %{z:.2f}%<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family=DEFAULT_FONT_FAMILY),
            x=0.5,
        ),
        template=DEFAULT_TEMPLATE,
        font=dict(family=DEFAULT_FONT_FAMILY, size=DEFAULT_FONT_SIZE),
        width=width,
        height=height,
        xaxis_title="Месяц",
        yaxis_title="Год",
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


def plot_backtest_dashboard(
    result: BacktestResult,
    benchmark: Optional[pd.Series] = None,
    width: int = 1400,
    height: int = 1600,
) -> go.Figure:
    """
    Сводный дашборд по результатам бэктеста.

    Включает:
    1. Equity curve
    2. Drawdown
    3. Daily PnL
    4. Месячные доходности (heatmap)
    5. Ключевые метрики (annotations)
    """

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        row_heights=[0.35, 0.20, 0.20, 0.25],
        subplot_titles=[
            "Equity Curve",
            "Drawdown",
            "Daily PnL",
            "Месячные доходности (%)",
        ],
    )

    # 1. Equity curve
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index.tolist(),
        y=result.equity_curve.values.tolist(),
        mode="lines",
        name=result.strategy_name,
        line=dict(color=COLOR_EQUITY, width=2),
        showlegend=True,
    ), row=1, col=1)

    if benchmark is not None:
        common_idx = result.equity_curve.index.intersection(benchmark.index)
        fig.add_trace(go.Scatter(
            x=common_idx.tolist(),
            y=benchmark.loc[common_idx].values.tolist(),
            mode="lines",
            name="Benchmark",
            line=dict(color=COLOR_BENCHMARK, width=1.5, dash="dash"),
        ), row=1, col=1)

    # 2. Drawdown
    equity = result.equity_curve.values
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index.tolist(),
        y=drawdown.tolist(),
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color=COLOR_DRAWDOWN, width=1),
        fillcolor="rgba(214, 39, 40, 0.3)",
        showlegend=False,
    ), row=2, col=1)

    # 3. Daily PnL
    pnl = result.daily_pnl["net"]
    colors = [
        COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE
        for v in pnl.values
    ]
    fig.add_trace(go.Bar(
        x=pnl.index.tolist(),
        y=pnl.values.tolist(),
        marker_color=colors,
        name="Daily PnL",
        showlegend=False,
    ), row=3, col=1)

    # 4. Monthly returns heatmap (как отдельный subplot не получится —
    #    используем annotations вместо heatmap)
    monthly = result.equity_curve.resample("M").last().pct_change().dropna() * 100
    monthly_by_year = monthly.groupby(
        [monthly.index.year, monthly.index.month]
    ).first().unstack()
    monthly_by_year = monthly_by_year.reindex(columns=range(1, 13))

    # Аннотации для месячных доходностей
    ru_months = [
        "Янв", "Фев", "Мар", "Апр", "Май", "Июн",
        "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек",
    ]

    # Заголовки месяцев
    for i, m in enumerate(ru_months):
        fig.add_annotation(
            text=f"<b>{m}</b>",
            xref="x4", yref="y4",
            x=i, y=len(monthly_by_year),
            showarrow=False,
            font=dict(size=10),
            row=4, col=1,
        )

    # Значения по годам
    for y_idx, year in enumerate(monthly_by_year.index):
        fig.add_annotation(
            text=f"<b>{year}</b>",
            xref="x4", yref="y4",
            x=-0.5, y=len(monthly_by_year) - y_idx - 1,
            showarrow=False,
            font=dict(size=10),
            row=4, col=1,
        )
        for m_idx, m in enumerate(range(1, 13)):
            val = monthly_by_year.loc[year, m] if m in monthly_by_year.columns else np.nan
            if pd.isna(val):
                text = "—"
                color = "#cccccc"
            else:
                text = f"{val:.1f}"
                color = COLOR_POSITIVE if val > 0 else COLOR_NEGATIVE if val < 0 else "#cccccc"
            fig.add_annotation(
                text=text,
                xref="x4", yref="y4",
                x=m_idx, y=len(monthly_by_year) - y_idx - 1,
                showarrow=False,
                font=dict(size=9, color=color),
                row=4, col=1,
            )

    # 5. Аннотация с ключевыми метриками
    s = result.summary()
    metrics_text = (
        f"Total: {s['total_return_pct']:+.2f}%  |  "
        f"Ann. return: {s['annualized_return'] * 100:+.2f}%  |  "
        f"Sharpe: {s['sharpe_ratio']:.2f}  |  "
        f"Sortino: {s['sortino_ratio']:.2f}  |  "
        f"MaxDD: {s['max_drawdown'] * 100:.2f}%  |  "
        f"Win rate: {s['win_rate'] * 100:.1f}%"
    )

    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=0.5, y=1.02,
        showarrow=False,
        font=dict(size=11, family=DEFAULT_FONT_FAMILY),
    )

    fig.update_layout(
        title=dict(
            text=f"Backtest Dashboard — {result.strategy_name.upper()}",
            font=dict(size=18, family=DEFAULT_FONT_FAMILY),
            x=0.5,
        ),
        template=DEFAULT_TEMPLATE,
        font=dict(family=DEFAULT_FONT_FAMILY, size=DEFAULT_FONT_SIZE),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=40, t=100, b=40),
    )

    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="DD, %", row=2, col=1)
    fig.update_yaxes(title_text="PnL", row=3, col=1)

    return fig


###############################################################################
# Convenience wrappers
###############################################################################

def quick_backtest(
    interpolator: Any,
    actual: pd.DataFrame,
    strategy: str = "mean_reversion",
    **kwargs: Any,
) -> BacktestResult:
    """
    Быстрый запуск бэктеста обученного интерполятора.

    Parameters
    ----------
    interpolator
        Обученный YieldCurveInterpolator.
    actual
        Эталонная кривая.
    strategy
        Имя стратегии.
    **kwargs
        Параметры для CurveBacktester.

    Returns
    -------
    BacktestResult
    """

    if not getattr(interpolator, "is_fitted_", False):
        raise RuntimeError("Интерполятор не обучен.")

    bt = CurveBacktester(strategy=strategy, **kwargs)
    return bt.run(
        predicted=interpolator.curve_,
        actual=actual,
    )
