"""
yield_curve.diagnostics
=======================

Модуль диагностики качества моделей восстановления
рыночной кривой депозитных ставок.

Функциональность
----------------
- Проверка сходимости EM-алгоритма
- Анализ объяснённой дисперсии и выбор размерности
- Анализ остатков (нормальность, автокорреляция)
- Поиск выбросов и точек с высоким рычагом (leverage)
- Сводный диагностический отчёт

Автор:
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats


###############################################################################
# Constants
###############################################################################

DEFAULT_OUTLIER_THRESHOLD = 3.0
DEFAULT_VARIANCE_THRESHOLD = 0.95
DEFAULT_LB_LAGS = 10
MIN_OBSERVATIONS_FOR_TESTS = 8


###############################################################################
# Convergence diagnostics
###############################################################################

def check_convergence(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Проверка сходимости EM-алгоритма по сводке движка.

    Parameters
    ----------
    summary
        Словарь, возвращаемый методом engine.summary().
        Ожидается наличие ключей: 'converged', 'n_iter', 'final_error',
        'tolerance'.

    Returns
    -------
    dict
        Словарь с результатами проверки:
        - converged: bool
        - n_iter: int
        - final_error: float
        - tolerance: float
        - margin: float (насколько финальная ошибка меньше tolerance)
        - verdict: str ('OK', 'WARNING', 'FAIL')
    """

    converged = bool(summary.get("converged", False))
    n_iter = int(summary.get("n_iter", 0))
    final_error = float(summary.get("final_error", np.inf))
    tolerance = float(summary.get("tolerance", 0.0))

    if tolerance > 0:
        margin = tolerance - final_error
    else:
        margin = 0.0

    if not converged:
        verdict = "FAIL"
    elif final_error > tolerance * 10:
        verdict = "WARNING"
    else:
        verdict = "OK"

    return {
        "converged": converged,
        "n_iter": n_iter,
        "final_error": final_error,
        "tolerance": tolerance,
        "margin": margin,
        "verdict": verdict,
    }


def convergence_history(
    engine: Any,
) -> Optional[pd.DataFrame]:
    """
    Извлечение истории ошибок по итерациям из движка.

    Parameters
    ----------
    engine
        Обученный движок (например, EMPCAEngine).

    Returns
    -------
    DataFrame или None
        DataFrame с колонками ['iter', 'error'].
        None, если движок не хранит историю.
    """

    history = getattr(engine, "error_history_", None)

    if history is None:
        return None

    if isinstance(history, pd.DataFrame):
        return history.copy()

    history_arr = np.asarray(history)

    if history_arr.ndim == 1:
        history_arr = history_arr.reshape(-1, 1)

    if history_arr.shape[1] == 1:
        columns = ["error"]
    else:
        columns = [f"error_{i}" for i in range(history_arr.shape[1])]

    return pd.DataFrame(
        history_arr,
        columns=columns,
    ).reset_index().rename(columns={"index": "iter"})


###############################################################################
# Variance explained
###############################################################################

def _total_variance(original: pd.DataFrame) -> float:
    """Общая дисперсия наблюдаемых точек."""
    return float(np.nanvar(original.values))


def explained_variance(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> pd.Series:
    """
    Доля дисперсии, объяснённая каждой компонентой.

    Вычисляется как отношение дисперсии проекции на k-ю компоненту
    к общей дисперсии наблюдаемых данных.

    Parameters
    ----------
    original
        Исходная матрица ставок (wide).
    reconstructed
        Реконструированная матрица ставок (wide).

    Returns
    -------
    Series
        Индекс — номер компоненты (0, 1, 2, ...),
        значения — доля объяснённой дисперсии.
    """

    total_var = _total_variance(original)

    if total_var == 0 or np.isnan(total_var):
        return pd.Series(dtype=float, name="explained_variance")

    X = original.values.astype(float)
    X_hat = reconstructed.values.astype(float)

    # Центрируем по наблюдаемым точкам
    mean_obs = np.nanmean(X)
    X_c = X - mean_obs
    X_hat_c = X_hat - mean_obs

    # Остаточная дисперсия
    residual_var = float(np.nanmean((X_c - X_hat_c) ** 2))
    explained = 1.0 - residual_var / total_var

    # Раскладываем по компонентам через последовательное вычитание
    components = getattr(reconstructed, "_components_decomposition_", None)

    if components is None:
        return pd.Series(
            [max(0.0, explained)],
            index=[0],
            name="explained_variance",
        )

    ratios = []
    cumulative = 0.0

    for comp in components:
        comp_var = float(np.nanmean((comp - np.nanmean(comp)) ** 2))
        ratio = comp_var / total_var
        cumulative += ratio
        ratios.append(min(max(ratio, 0.0), 1.0))

    return pd.Series(
        ratios,
        index=list(range(len(ratios))),
        name="explained_variance",
    )


def cumulative_variance(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> pd.Series:
    """
    Кумулятивная доля объяснённой дисперсии.
    """

    ev = explained_variance(original, reconstructed)

    if ev.empty:
        return ev

    cumulative = ev.cumsum()
    cumulative.name = "cumulative_variance"

    return cumulative


def optimal_n_components(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
    threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> int:
    """
    Рекомендация числа компонент по порогу объяснённой дисперсии.

    Parameters
    ----------
    original
    reconstructed
    threshold
        Порог кумулятивной объяснённой дисперсии (по умолчанию 0.95).

    Returns
    -------
    int
        Минимальное число компонент, при котором кумулятивная
        объяснённая дисперсия >= threshold.
    """

    cv = cumulative_variance(original, reconstructed)

    if cv.empty:
        return 1

    above = cv[cv >= threshold]

    if above.empty:
        return len(cv)

    return int(above.index[0]) + 1


###############################################################################
# Residual analysis
###############################################################################

def compute_residuals(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Вычисление остатков (наблюдаемое - реконструкция).

    Parameters
    ----------
    original
        Исходная матрица ставок.
    reconstructed
        Реконструированная матрица.
    mask
        Булева маска пропусков. Если передана, остатки по пропущенным
        точкам будут заменены на NaN (нас интересует только качество
        реконструкции наблюдаемых данных).

    Returns
    -------
    DataFrame
        Матрица остатков той же формы, что и original.
    """

    residuals = original - reconstructed

    if mask is not None:
        residuals = residuals.mask(mask, np.nan)

    return residuals


def residual_summary(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Сводная статистика остатков по бакетам (колонкам).

    Returns
    -------
    DataFrame
        Индекс — бакеты, колонки:
        ['mean', 'std', 'mae', 'rmse', 'skew', 'kurt', 'n_obs'].
    """

    records = []

    for col in residuals.columns:
        series = residuals[col].dropna()
        n = len(series)

        if n == 0:
            records.append({
                "bucket": col,
                "mean": np.nan,
                "std": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "skew": np.nan,
                "kurt": np.nan,
                "n_obs": 0,
            })
            continue

        mean = float(series.mean())
        std = float(series.std(ddof=1)) if n > 1 else 0.0
        mae = float(series.abs().mean())
        rmse = float(np.sqrt((series ** 2).mean()))
        skew = float(stats.skew(series, bias=False)) if n >= 3 else np.nan
        kurt = float(stats.kurtosis(series, bias=False)) if n >= 4 else np.nan

        records.append({
            "bucket": col,
            "mean": mean,
            "std": std,
            "mae": mae,
            "rmse": rmse,
            "skew": skew,
            "kurt": kurt,
            "n_obs": n,
        })

    return pd.DataFrame(records).set_index("bucket")


def test_normality(
    residuals: pd.DataFrame,
    method: str = "jarque_bera",
) -> pd.DataFrame:
    """
    Тест нормальности остатков по бакетам.

    Parameters
    ----------
    residuals
    method
        'jarque_bera' (быстрый, асимптотический)
        или 'shapiro' (мощный, для малых выборок).

    Returns
    -------
    DataFrame
        Индекс — бакеты, колонки:
        ['statistic', 'p_value', 'is_normal', 'n_obs'].
        is_normal: p_value > 0.05.
    """

    records = []

    for col in residuals.columns:
        series = residuals[col].dropna().values
        n = len(series)

        if n < MIN_OBSERVATIONS_FOR_TESTS:
            records.append({
                "bucket": col,
                "statistic": np.nan,
                "p_value": np.nan,
                "is_normal": np.nan,
                "n_obs": n,
            })
            continue

        if method == "jarque_bera":
            stat, pval = stats.jarque_bera(series)
        elif method == "shapiro":
            if n > 5000:
                series = np.random.RandomState(42).choice(
                    series, size=5000, replace=False
                )
            stat, pval = stats.shapiro(series)
        else:
            raise ValueError(f"Unknown normality test: {method}")

        records.append({
            "bucket": col,
            "statistic": float(stat),
            "p_value": float(pval),
            "is_normal": bool(pval > 0.05),
            "n_obs": n,
        })

    return pd.DataFrame(records).set_index("bucket")


def test_autocorrelation(
    residuals: pd.DataFrame,
    lags: int = DEFAULT_LB_LAGS,
) -> pd.DataFrame:
    """
    Тест Льюнга-Бокса на отсутствие автокорреляции остатков.

    Нулевая гипотеза: остатки — белый шум (автокорреляции нет).

    Parameters
    ----------
    residuals
    lags
        Число лагов для теста.

    Returns
    -------
    DataFrame
        Индекс — бакеты, колонки:
        ['statistic', 'p_value', 'is_white_noise', 'n_obs'].
        is_white_noise: p_value > 0.05.
    """

    records = []

    for col in residuals.columns:
        series = residuals[col].dropna().values
        n = len(series)

        if n < lags + MIN_OBSERVATIONS_FOR_TESTS:
            records.append({
                "bucket": col,
                "statistic": np.nan,
                "p_value": np.nan,
                "is_white_noise": np.nan,
                "n_obs": n,
            })
            continue

        # Реализация Ljung-Box через автокорреляции
        series_c = series - series.mean()
        var = np.sum(series_c ** 2)

        if var == 0:
            records.append({
                "bucket": col,
                "statistic": 0.0,
                "p_value": 1.0,
                "is_white_noise": True,
                "n_obs": n,
            })
            continue

        lb_stat = 0.0
        for k in range(1, lags + 1):
            rho = np.sum(series_c[k:] * series_c[:-k]) / var
            lb_stat += (rho ** 2) / (n - k)

        lb_stat *= n * (n + 2)
        pval = 1.0 - stats.chi2.cdf(lb_stat, df=lags)

        records.append({
            "bucket": col,
            "statistic": float(lb_stat),
            "p_value": float(pval),
            "is_white_noise": bool(pval > 0.05),
            "n_obs": n,
        })

    return pd.DataFrame(records).set_index("bucket")


###############################################################################
# Outlier and leverage detection
###############################################################################

def find_outliers(
    residuals: pd.DataFrame,
    threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    method: str = "zscore",
) -> pd.DataFrame:
    """
    Поиск выбросов в остатках.

    Parameters
    ----------
    residuals
    threshold
        Порог отсечки.
    method
        'zscore' — стандартные отклонения.
        'mad' — медианное абсолютное отклонение (устойчивее к самим выбросам).

    Returns
    -------
    DataFrame
        Таблица с колонками:
        ['date', 'bucket', 'residual', 'score', 'is_outlier'].
    """

    records = []

    for col in residuals.columns:
        series = residuals[col].dropna()

        if series.empty:
            continue

        if method == "zscore":
            mean = series.mean()
            scale = series.std(ddof=1)
            if scale == 0 or np.isnan(scale):
                continue
            scores = (series - mean) / scale
        elif method == "mad":
            median = series.median()
            mad = np.median(np.abs(series - median))
            if mad == 0:
                continue
            scores = 0.6745 * (series - median) / mad
        else:
            raise ValueError(f"Unknown outlier method: {method}")

        for date, value, score in zip(series.index, series.values, scores.values):
            records.append({
                "date": date,
                "bucket": col,
                "residual": float(value),
                "score": float(score),
                "is_outlier": bool(abs(score) > threshold),
            })

    return pd.DataFrame(records)


def leverage_by_bucket(
    residuals: pd.DataFrame,
) -> pd.Series:
    """
    Средний абсолютный остаток по бакетам.

    Показывает, какие бакеты модель восстанавливает хуже всего.
    """

    return residuals.abs().mean(axis=0).rename("mean_abs_residual")


def leverage_by_date(
    residuals: pd.DataFrame,
) -> pd.Series:
    """
    Средний абсолютный остаток по датам.

    Показывает, в какие даты кривая восстанавливается хуже всего
    (например, дни аномальной волатильности или праздники).
    """

    return residuals.abs().mean(axis=1).rename("mean_abs_residual")


###############################################################################
# Summary report
###############################################################################

def full_report(
    interpolator: Any,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> Dict[str, Any]:
    """
    Сводный диагностический отчёт по обученному интерполятору.

    Parameters
    ----------
    interpolator
        Обученный YieldCurveInterpolator (после вызова fit()).
    variance_threshold
        Порог для optimal_n_components.
    outlier_threshold
        Порог для поиска выбросов.

    Returns
    -------
    dict
        Словарь с ключами:
        - 'convergence': dict (результат check_convergence)
        - 'convergence_history': DataFrame или None
        - 'explained_variance': Series
        - 'cumulative_variance': Series
        - 'optimal_n_components': int
        - 'residual_summary': DataFrame
        - 'normality_jb': DataFrame
        - 'autocorrelation_lb': DataFrame
        - 'outliers': DataFrame
        - 'leverage_by_bucket': Series
        - 'leverage_by_date': Series
    """

    if not getattr(interpolator, "is_fitted_", False):
        raise RuntimeError(
            "Интерполятор не обучен. Сначала вызовите fit()."
        )

    original = interpolator.original_curve_
    reconstructed = interpolator.curve_
    mask = interpolator.mask_
    engine = interpolator._engine

    residuals = compute_residuals(original, reconstructed, mask)

    report: Dict[str, Any] = {}

    # 1. Сходимость
    try:
        summary = engine.summary()
        report["convergence"] = check_convergence(summary)
    except Exception:
        report["convergence"] = None

    report["convergence_history"] = convergence_history(engine)

    # 2. Дисперсия
    report["explained_variance"] = explained_variance(original, reconstructed)
    report["cumulative_variance"] = cumulative_variance(original, reconstructed)
    report["optimal_n_components"] = optimal_n_components(
        original, reconstructed, variance_threshold
    )

    # 3. Остатки
    report["residual_summary"] = residual_summary(residuals)
    report["normality_jb"] = test_normality(residuals, method="jarque_bera")
    report["autocorrelation_lb"] = test_autocorrelation(residuals)

    # 4. Выбросы и рычаг
    report["outliers"] = find_outliers(residuals, outlier_threshold)
    report["leverage_by_bucket"] = leverage_by_bucket(residuals)
    report["leverage_by_date"] = leverage_by_date(residuals)

    return report


def print_report(
    report: Dict[str, Any],
) -> None:
    """
    Печать сводного отчёта в читаемом виде.
    """

    print("=" * 70)
    print("YIELD CURVE DIAGNOSTICS REPORT")
    print("=" * 70)

    # Convergence
    conv = report.get("convergence")
    print("\n[1] CONVERGENCE")
    if conv is None:
        print("    No convergence info available.")
    else:
        print(f"    Converged:    {conv['converged']}")
        print(f"    Iterations:   {conv['n_iter']}")
        print(f"    Final error:  {conv['final_error']:.6e}")
        print(f"    Tolerance:    {conv['tolerance']:.6e}")
        print(f"    Verdict:      {conv['verdict']}")

    # Variance
    print("\n[2] VARIANCE EXPLAINED")
    cv = report.get("cumulative_variance")
    if cv is not None and not cv.empty:
        for i, v in cv.items():
            print(f"    PC{i}: {v:.4f}")
    print(f"    Optimal n_components (threshold=0.95): "
          f"{report.get('optimal_n_components', 'N/A')}")

    # Residuals
    print("\n[3] RESIDUAL SUMMARY (by bucket)")
    rs = report.get("residual_summary")
    if rs is not None and not rs.empty:
        print(rs.to_string())

    # Normality
    print("\n[4] NORMALITY TEST (Jarque-Bera)")
    norm = report.get("normality_jb")
    if norm is not None and not norm.empty:
        n_normal = norm["is_normal"].sum()
        n_total = norm["is_normal"].notna().sum()
        print(f"    Normal buckets: {n_normal} / {n_total}")

    # Autocorrelation
    print("\n[5] AUTOCORRELATION TEST (Ljung-Box)")
    ac = report.get("autocorrelation_lb")
    if ac is not None and not ac.empty:
        n_white = ac["is_white_noise"].sum()
        n_total = ac["is_white_noise"].notna().sum()
        print(f"    White-noise buckets: {n_white} / {n_total}")

    # Outliers
    print("\n[6] OUTLIERS")
    outliers = report.get("outliers")
    if outliers is not None and not outliers.empty:
        n_out = outliers["is_outlier"].sum()
        print(f"    Total outliers detected: {n_out}")
        if n_out > 0:
            print("    Top-5 by |score|:")
            top = (
                outliers[outliers["is_outlier"]]
                .reindex(outliers["score"].abs().sort_values(ascending=False).index)
                .head(5)
            )
            print(top.to_string(index=False))
    else:
        print("    No outliers.")

    # Leverage
    print("\n[7] LEVERAGE BY BUCKET (worst 3)")
    lb = report.get("leverage_by_bucket")
    if lb is not None and not lb.empty:
        worst = lb.sort_values(ascending=False).head(3)
        for bucket, val in worst.items():
            print(f"    bucket {bucket}: {val:.6f}")

    print("\n" + "=" * 70)
