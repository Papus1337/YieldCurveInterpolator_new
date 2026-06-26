"""
yield_curve.metrics
===================

Модуль оценки качества восстановления кривой ставок
путём сравнения с эталонной кривой.

Функциональность
----------------
- Point-wise метрики (RMSE, MAE, MAPE, Max Error)
- Направленная точность (directional accuracy)
- Кривые метрики (L2-норма, косинусное сходство формы)
- Ранговые метрики (Spearman correlation)
- Экономические метрики (PnL от торговли на сигналах)
- Агрегация по бакетам и датам
- Сводный отчёт по сравнению двух кривых

Автор:
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats


###############################################################################
# Constants
###############################################################################

DEFAULT_DURATION_BY_BUCKET = {
    1: 1 / 365,
    7: 7 / 365,
    14: 14 / 365,
    31: 31 / 365,
    61: 61 / 365,
    91: 91 / 365,
    181: 181 / 365,
}

DEFAULT_NOTIONAL = 1_000_000.0
DEFAULT_BPS = 100.0


###############################################################################
# Alignment helpers
###############################################################################

def _align_curves(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Приведение предсказанной и эталонной кривых к общему индексу и колонкам.

    Оставляет только пересечение дат и бакетов, где обе кривые имеют
    валидные (не-NaN) значения.

    Parameters
    ----------
    predicted
        Реконструированная кривая (wide).
    actual
        Эталонная кривая (wide).
    mask
        Дополнительная маска пропусков. Если передана, точки,
        где mask == True, будут исключены из сравнения.

    Returns
    -------
    tuple
        (aligned_predicted, aligned_actual, aligned_mask)
    """

    common_idx = predicted.index.intersection(actual.index)
    common_cols = predicted.columns.intersection(actual.columns)

    p = predicted.loc[common_idx, common_cols].copy()
    a = actual.loc[common_idx, common_cols].copy()

    # Исключаем точки, где хотя бы одно из значений NaN
    valid = p.notna() & a.notna()
    p = p.where(valid)
    a = a.where(valid)

    aligned_mask = None
    if mask is not None:
        m = mask.loc[common_idx, common_cols].copy()
        m = m.where(valid, False)
        aligned_mask = m

    return p, a, aligned_mask


def _apply_mask(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Применение маски: точки, где mask == True, зануляются."""
    if mask is None:
        return predicted, actual

    p = predicted.mask(mask)
    a = actual.mask(mask)
    return p, a


###############################################################################
# Point-wise metrics
###############################################################################

def rmse(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> float:
    """
    Root Mean Squared Error между предсказанной и эталонной кривыми.

    Измеряется в тех же единицах, что и ставки (например, % годовых).
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    diff = (p - a).values.flatten()
    diff = diff[~np.isnan(diff)]

    if diff.size == 0:
        return float("nan")

    return float(np.sqrt(np.mean(diff ** 2)))


def mae(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> float:
    """
    Mean Absolute Error.

    Более робастная метрика, чем RMSE — меньше чувствительна к выбросам.
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    diff = (p - a).values.flatten()
    diff = diff[~np.isnan(diff)]

    if diff.size == 0:
        return float("nan")

    return float(np.mean(np.abs(diff)))


def mape(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    epsilon: float = 1e-8,
) -> float:
    """
    Mean Absolute Percentage Error (в процентах).

    Внимание: метрика не определена, если эталонные значения близки к нулю.
    Для ставок это обычно не проблема (ставки > 0).
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    p_vals = p.values.flatten()
    a_vals = a.values.flatten()

    valid = ~(np.isnan(p_vals) | np.isnan(a_vals) | (np.abs(a_vals) < epsilon))
    if not valid.any():
        return float("nan")

    p_v = p_vals[valid]
    a_v = a_vals[valid]

    return float(np.mean(np.abs((a_v - p_v) / a_v)) * 100.0)


def max_error(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> float:
    """
    Максимальная абсолютная ошибка.

    Показывает худший случай — полезно для оценки tail-risk модели.
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    diff = (p - a).values.flatten()
    diff = diff[~np.isnan(diff)]

    if diff.size == 0:
        return float("nan")

    return float(np.max(np.abs(diff)))


def pointwise_metrics(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Сводка всех point-wise метрик.
    """

    return {
        "rmse": rmse(predicted, actual, mask),
        "mae": mae(predicted, actual, mask),
        "mape": mape(predicted, actual, mask),
        "max_error": max_error(predicted, actual, mask),
    }


###############################################################################
# Directional accuracy
###############################################################################

def directional_accuracy(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> float:
    """
    Доля случаев, когда модель правильно предсказала направление
    изменения ставки (вверх/вниз) от предыдущей даты.

    Критически важная метрика для трейдинговых стратегий:
    даже если RMSE высок, но направление угадано верно,
    на сигналах модели можно зарабатывать.

    Returns
    -------
    float
        Доля верных предсказаний направления (от 0 до 1).
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    p_diff = p.diff(axis=0)
    a_diff = a.diff(axis=0)

    # Знак изменения
    p_sign = np.sign(p_diff.values.flatten())
    a_sign = np.sign(a_diff.values.flatten())

    valid = ~(np.isnan(p_sign) | np.isnan(a_sign) | (a_sign == 0))
    if not valid.any():
        return float("nan")

    p_s = p_sign[valid]
    a_s = a_sign[valid]

    correct = (p_s == a_s).sum()
    return float(correct / len(a_s))


def directional_accuracy_by_bucket(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Направленная точность по каждому бакету.
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    p_diff = p.diff(axis=0)
    a_diff = a.diff(axis=0)

    results = {}
    for col in p.columns:
        p_s = np.sign(p_diff[col].values)
        a_s = np.sign(a_diff[col].values)
        valid = ~(np.isnan(p_s) | np.isnan(a_s) | (a_s == 0))
        if not valid.any():
            results[col] = float("nan")
            continue
        correct = (p_s[valid] == a_s[valid]).sum()
        results[col] = float(correct / valid.sum())

    return pd.Series(results, name="directional_accuracy")


###############################################################################
# Curve-level metrics
###############################################################################

def curve_l2_distance(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    L2-норма разности кривых по каждой дате.

    Показывает, насколько сильно предсказанная кривая
    отклоняется от эталонной как единый объект.
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    diff = p - a
    l2 = np.sqrt((diff ** 2).sum(axis=1))
    return l2.rename("l2_distance")


def curve_cosine_similarity(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Косинусное сходство формы кривых по каждой дате.

    Метрика инвариантна к уровню — показывает, насколько
    похожа форма (наклон, выпуклость), игнорируя сдвиг.
    Значения от -1 до 1, где 1 — полное совпадение формы.
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    results = {}
    for date in p.index:
        p_vec = p.loc[date].values
        a_vec = a.loc[date].values

        valid = ~(np.isnan(p_vec) | np.isnan(a_vec))
        if valid.sum() < 2:
            results[date] = float("nan")
            continue

        p_v = p_vec[valid]
        a_v = a_vec[valid]

        norm_p = np.linalg.norm(p_v)
        norm_a = np.linalg.norm(a_v)

        if norm_p == 0 or norm_a == 0:
            results[date] = float("nan")
            continue

        cos_sim = float(np.dot(p_v, a_v) / (norm_p * norm_a))
        results[date] = cos_sim

    return pd.Series(results, name="cosine_similarity")


###############################################################################
# Rank-based metrics
###############################################################################

def rank_correlation(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    method: str = "spearman",
) -> pd.Series:
    """
    Ранговая корреляция между предсказанной и эталонной кривыми
    по каждой дате.

    Показывает, правильно ли модель ранжирует бакеты
    (короткий конец дешевле длинного, или наоборот —
    в случае инверсии).

    Parameters
    ----------
    method
        'spearman' или 'kendall'.

    Returns
    -------
    Series
        Индекс — даты, значения — коэффициент корреляции.
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    results = {}
    for date in p.index:
        p_vec = p.loc[date].values
        a_vec = a.loc[date].values

        valid = ~(np.isnan(p_vec) | np.isnan(a_vec))
        if valid.sum() < 3:
            results[date] = float("nan")
            continue

        p_v = p_vec[valid]
        a_v = a_vec[valid]

        if method == "spearman":
            corr, _ = stats.spearmanr(p_v, a_v)
        elif method == "kendall":
            corr, _ = stats.kendalltau(p_v, a_v)
        else:
            raise ValueError(f"Unknown rank method: {method}")

        results[date] = float(corr)

    return pd.Series(results, name=f"{method}_correlation")


###############################################################################
# Economic metrics
###############################################################################

def pnl_per_bucket(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    notional: float = DEFAULT_NOTIONAL,
    duration_map: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Гипотетический PnL от торговли на сигналах модели по каждому бакету.

    Логика:
    - Если модель предсказывает ставку НИЖЕ эталонной, мы "занимаем"
      по предсказанной и "размещаем" по эталонной — получаем доход.
    - Если модель предсказывает ставку ВЫШЕ эталонной — наоборот.
    - PnL нормирован на duration и notional.

    Это упрощённая модель, но она даёт экономическую интерпретацию
    качества интерполяции.

    Parameters
    ----------
    predicted
    actual
    mask
    notional
        Нотионал в единицах (по умолчанию 1 млн).
    duration_map
        Словарь {bucket: duration_in_years}. Если не передан,
        используется DEFAULT_DURATION_BY_BUCKET.

    Returns
    -------
    DataFrame
        Индекс — даты, колонки — бакеты, значения — PnL в единицах.
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    if duration_map is None:
        duration_map = DEFAULT_DURATION_BY_BUCKET

    # Разница в ставках (в процентных пунктах)
    spread = (a - p) / DEFAULT_BPS

    pnl = pd.DataFrame(index=p.index, columns=p.columns, dtype=float)

    for col in p.columns:
        bucket = int(col) if not isinstance(col, int) else col
        duration = duration_map.get(bucket, bucket / 365.0)
        pnl[col] = spread[col] * duration * notional

    return pnl


def total_pnl(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    notional: float = DEFAULT_NOTIONAL,
    duration_map: Optional[Dict[int, float]] = None,
) -> float:
    """
    Суммарный PnL по всем датам и бакетам.
    """

    pnl = pnl_per_bucket(predicted, actual, mask, notional, duration_map)
    return float(pnl.sum().sum())


def annualized_pnl(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    notional: float = DEFAULT_NOTIONAL,
    duration_map: Optional[Dict[int, float]] = None,
    trading_days_per_year: int = 252,
) -> float:
    """
    PnL, приведённый к годовой ставке.
    """

    pnl = pnl_per_bucket(predicted, actual, mask, notional, duration_map)
    n_days = len(pnl)
    if n_days == 0:
        return float("nan")

    total = pnl.sum().sum()
    return float(total / n_days * trading_days_per_year)


###############################################################################
# Aggregation
###############################################################################

def metrics_by_bucket(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Сводка метрик по каждому бакету.

    Returns
    -------
    DataFrame
        Индекс — бакеты, колонки:
        ['rmse', 'mae', 'mape', 'max_error', 'directional_accuracy',
         'mean_bias', 'n_obs'].
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    records = []
    for col in p.columns:
        p_col = p[col].dropna()
        a_col = a[col].dropna()

        common_idx = p_col.index.intersection(a_col.index)
        if len(common_idx) == 0:
            records.append({
                "bucket": col,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "max_error": np.nan,
                "directional_accuracy": np.nan,
                "mean_bias": np.nan,
                "n_obs": 0,
            })
            continue

        p_v = p_col.loc[common_idx].values
        a_v = a_col.loc[common_idx].values
        diff = a_v - p_v

        n = len(diff)
        rmse_v = float(np.sqrt(np.mean(diff ** 2)))
        mae_v = float(np.mean(np.abs(diff)))
        max_err = float(np.max(np.abs(diff)))
        bias = float(np.mean(diff))

        valid_abs = np.abs(a_v) > 1e-8
        if valid_abs.any():
            mape_v = float(np.mean(np.abs(diff[valid_abs] / a_v[valid_abs])) * 100)
        else:
            mape_v = float("nan")

        # Directional accuracy
        p_diff = np.diff(p_v)
        a_diff = np.diff(a_v)
        valid_dir = ~(np.isnan(p_diff) | np.isnan(a_diff) | (a_diff == 0))
        if valid_dir.any():
            da = float((np.sign(p_diff[valid_dir]) == np.sign(a_diff[valid_dir])).mean())
        else:
            da = float("nan")

        records.append({
            "bucket": col,
            "rmse": rmse_v,
            "mae": mae_v,
            "mape": mape_v,
            "max_error": max_err,
            "directional_accuracy": da,
            "mean_bias": bias,
            "n_obs": n,
        })

    return pd.DataFrame(records).set_index("bucket")


def metrics_by_date(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Сводка метрик по каждой дате.

    Полезна для поиска дат, когда модель работает аномально плохо
    (например, дни аукционов с аномальным спросом).

    Returns
    -------
    DataFrame
        Индекс — даты, колонки:
        ['rmse', 'mae', 'max_error', 'l2_distance', 'cosine_similarity',
         'n_obs'].
    """

    p, a = _align_curves(predicted, actual, mask)
    p, a = _apply_mask(p, a, mask)

    records = []
    for date in p.index:
        p_row = p.loc[date].dropna()
        a_row = a.loc[date].dropna()

        common_cols = p_row.index.intersection(a_row.index)
        if len(common_cols) == 0:
            records.append({
                "date": date,
                "rmse": np.nan,
                "mae": np.nan,
                "max_error": np.nan,
                "l2_distance": np.nan,
                "cosine_similarity": np.nan,
                "n_obs": 0,
            })
            continue

        p_v = p_row.loc[common_cols].values
        a_v = a_row.loc[common_cols].values
        diff = a_v - p_v

        n = len(diff)
        rmse_v = float(np.sqrt(np.mean(diff ** 2)))
        mae_v = float(np.mean(np.abs(diff)))
        max_err = float(np.max(np.abs(diff)))
        l2 = float(np.sqrt(np.sum(diff ** 2)))

        norm_p = np.linalg.norm(p_v)
        norm_a = np.linalg.norm(a_v)
        if norm_p > 0 and norm_a > 0:
            cos_sim = float(np.dot(p_v, a_v) / (norm_p * norm_a))
        else:
            cos_sim = float("nan")

        records.append({
            "date": date,
            "rmse": rmse_v,
            "mae": mae_v,
            "max_error": max_err,
            "l2_distance": l2,
            "cosine_similarity": cos_sim,
            "n_obs": n,
        })

    return pd.DataFrame(records).set_index("date")


###############################################################################
# Summary report
###############################################################################

def compare_curves(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    notional: float = DEFAULT_NOTIONAL,
    duration_map: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Сводный отчёт по сравнению предсказанной и эталонной кривых.

    Parameters
    ----------
    predicted
        Реконструированная кривая (например, interpolator.curve_).
    actual
        Эталонная кривая (например, официальная кривая ЦБ).
    mask
        Маска пропусков (опционально).
    notional
        Нотионал для расчёта PnL.
    duration_map
        Словарь duration по бакетам.

    Returns
    -------
    dict
        Словарь с ключами:
        - 'pointwise': dict (rmse, mae, mape, max_error)
        - 'directional_accuracy': float
        - 'rank_correlation': Series
        - 'metrics_by_bucket': DataFrame
        - 'metrics_by_date': DataFrame
        - 'l2_distance': Series
        - 'cosine_similarity': Series
        - 'total_pnl': float
        - 'annualized_pnl': float
        - 'n_observations': int
    """

    p, a, aligned_mask = _align_curves(predicted, actual, mask)

    n_obs = int(p.notna().sum().sum())

    report: Dict[str, Any] = {}

    report["pointwise"] = pointwise_metrics(p, a, aligned_mask)
    report["directional_accuracy"] = directional_accuracy(p, a, aligned_mask)
    report["rank_correlation"] = rank_correlation(p, a, aligned_mask)
    report["metrics_by_bucket"] = metrics_by_bucket(p, a, aligned_mask)
    report["metrics_by_date"] = metrics_by_date(p, a, aligned_mask)
    report["l2_distance"] = curve_l2_distance(p, a, aligned_mask)
    report["cosine_similarity"] = curve_cosine_similarity(p, a, aligned_mask)
    report["total_pnl"] = total_pnl(p, a, aligned_mask, notional, duration_map)
    report["annualized_pnl"] = annualized_pnl(
        p, a, aligned_mask, notional, duration_map
    )
    report["n_observations"] = n_obs

    return report


def print_metrics_report(
    report: Dict[str, Any],
) -> None:
    """
    Печать сводного отчёта в читаемом виде.
    """

    print("=" * 70)
    print("YIELD CURVE METRICS REPORT")
    print("=" * 70)

    print(f"\nTotal observations: {report.get('n_observations', 'N/A')}")

    print("\n[1] POINTWISE METRICS")
    pw = report.get("pointwise", {})
    for k, v in pw.items():
        if k == "mape":
            print(f"    {k:12s}: {v:.4f}%")
        else:
            print(f"    {k:12s}: {v:.6f}")

    print(f"\n[2] DIRECTIONAL ACCURACY")
    da = report.get("directional_accuracy")
    if da is not None and not np.isnan(da):
        print(f"    {da:.4f} ({da * 100:.2f}%)")
    else:
        print("    N/A")

    print(f"\n[3] RANK CORRELATION (Spearman)")
    rc = report.get("rank_correlation")
    if rc is not None and not rc.empty:
        mean_rc = rc.mean()
        print(f"    Mean: {mean_rc:.4f}")
        print(f"    Min:  {rc.min():.4f}")
        print(f"    Max:  {rc.max():.4f}")

    print(f"\n[4] ECONOMIC METRICS")
    total = report.get("total_pnl")
    ann = report.get("annualized_pnl")
    if total is not None:
        print(f"    Total PnL:       {total:>15,.2f}")
    if ann is not None:
        print(f"    Annualized PnL:  {ann:>15,.2f}")

    print(f"\n[5] METRICS BY BUCKET (top 3 worst by RMSE)")
    mb = report.get("metrics_by_bucket")
    if mb is not None and not mb.empty:
        worst = mb.sort_values("rmse", ascending=False).head(3)
        print(worst.to_string())

    print(f"\n[6] METRICS BY DATE (top 5 worst by RMSE)")
    md = report.get("metrics_by_date")
    if md is not None and not md.empty:
        worst = md.sort_values("rmse", ascending=False).head(5)
        print(worst.to_string())

    print("\n" + "=" * 70)


###############################################################################
# Convenience wrappers for interpolator
###############################################################################

def evaluate_interpolator(
    interpolator: Any,
    actual: pd.DataFrame,
    notional: float = DEFAULT_NOTIONAL,
    duration_map: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Обёртка для оценки обученного интерполятора против эталона.

    Parameters
    ----------
    interpolator
        Обученный YieldCurveInterpolator.
    actual
        Эталонная кривая (wide-формат).
    notional
        Нотионал для PnL.
    duration_map
        Словарь duration по бакетам.

    Returns
    -------
    dict
        Результат compare_curves.
    """

    if not getattr(interpolator, "is_fitted_", False):
        raise RuntimeError(
            "Интерполятор не обучен. Сначала вызовите fit()."
        )

    predicted = interpolator.curve_
    mask = interpolator.mask_

    return compare_curves(
        predicted, actual, mask, notional, duration_map
    )
