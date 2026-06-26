"""
yield_curve.preprocessing
=========================

Модуль подготовки данных для моделей восстановления
рыночной кривой депозитных ставок.

Функциональность
----------------
- Проверка входных данных
- Построение матрицы ставок (с взвешиванием по объёму)
- Построение матрицы объёмов
- Создание маски пропусков
- Инициализация пропусков
- Масштабирование данных
- Обратное масштабирование
- Конвертация wide <-> long

Автор:
"""

from __future__ import annotations

from typing import Iterable
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


###############################################################################
# Constants
###############################################################################

DEFAULT_BUCKETS = (
    1,
    7,
    14,
    31,
    61,
    91,
    181,
)

REQUIRED_COLUMNS = {
    "date",
    "term_bucket",
    "rate",
    "amount",
}


###############################################################################
# Validation
###############################################################################

def validate_input(
    df: pd.DataFrame,
    buckets: Iterable[int] = DEFAULT_BUCKETS,
) -> None:
    """
    Проверка входного DataFrame.

    Parameters
    ----------
    df
        Исходный DataFrame.

    buckets
        Допустимые бакеты срочности.

    Raises
    ------
    TypeError
        Если объект не является DataFrame.

    ValueError
        Если отсутствуют необходимые столбцы
        либо обнаружены неизвестные term_bucket.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be pandas.DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:

        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    unknown = (
        set(df["term_bucket"].unique())
        - set(buckets)
    )

    if unknown:

        raise ValueError(
            f"Unknown term buckets: {sorted(unknown)}"
        )


###############################################################################
# Matrix builders
###############################################################################

def build_rate_matrix(
    df: pd.DataFrame,
    buckets: Iterable[int] = DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """
    Построение матрицы ставок (взвешенных по объёму amount).

    Логика взвешивания:
    1. Если по сделке есть rate и amount > 0, учитываем её вес.
    2. Если amount == NaN, заполняем средним объёмом по данному бакету, 
       чтобы сделка не была полностью проигнорирована.
    3. Если amount == 0 (или все amount в группе нулевые), 
       возвращаем простое среднее по rate, чтобы не терять данные.
    4. Если все rate в группе NaN, возвращаем NaN.

    Returns
    -------
    DataFrame

        index   -> date

        columns -> maturity

        values  -> weighted rate
    """

    df_work = df.copy()
    
    # Заполняем пропуски в amount средним значением по соответствующему бакету.
    # Это позволяет сделкам с неизвестным объёмом всё равно участвовать во взвешивании.
    df_work["amount"] = (
        df_work
        .groupby("term_bucket")["amount"]
        .transform(lambda x: x.fillna(x.mean()))
    )
    # Если во всём бакете объёмы были NaN, заполняем оставшиеся нулями
    df_work["amount"] = df_work["amount"].fillna(0.0)
    
    # Маска валидных (не-NaN) ставок
    valid_rate = df_work["rate"].notna()
    
    # Знаменатель: сумма amount только там, где rate валиден
    df_work["valid_amount"] = np.where(valid_rate, df_work["amount"], 0.0)
    
    # Числитель: сумма (rate * amount)
    # Если rate NaN, произведение тоже NaN. При суммировании pandas пропустит NaN.
    df_work["weighted_rate_sum"] = df_work["rate"] * df_work["amount"]
    
    # Агрегируем через pivot_table (работает быстрее, чем groupby.apply)
    num = df_work.pivot_table(
        index="date", 
        columns="term_bucket", 
        values="weighted_rate_sum", 
        aggfunc="sum"
    )
    
    den = df_work.pivot_table(
        index="date", 
        columns="term_bucket", 
        values="valid_amount", 
        aggfunc="sum"
    )
    
    # Считаем количество валидных rate в каждой группе
    count = df_work.pivot_table(
        index="date", 
        columns="term_bucket",
        values="rate", 
        aggfunc="count"
    )
    
    # Делим числитель на знаменатель
    curve = num / den
    
    # Если count == 0, значит не было ни одной валидной ставки -> принудительно NaN
    curve = curve.where(count > 0, np.nan)
    
    # Обработка edge-case: если count > 0, но все amount оказались нулевыми (den == 0)
    # В таком случае fallback на простое среднее, чтобы не терять информацию
    simple_mean = df_work.pivot_table(
        index="date", 
        columns="term_bucket",
        values="rate", 
        aggfunc="mean"
    )
    
    fallback_mask = (den == 0) & (count > 0)
    curve = curve.mask(fallback_mask, simple_mean)
    
    # Финальное выравнивание структуры
    curve = curve.reindex(columns=buckets).sort_index()
    curve.index = pd.to_datetime(curve.index)
    
    return curve


def build_amount_matrix(
    df: pd.DataFrame,
    buckets: Iterable[int] = DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """
    Построение матрицы объёмов.
    """

    amount = (
        df
        .pivot_table(
            index="date",
            columns="term_bucket",
            values="amount",
            aggfunc="sum",
        )
        .sort_index()
    )

    amount = amount.reindex(columns=buckets)

    amount.index = pd.to_datetime(amount.index)

    return amount


###############################################################################
# Missing values
###############################################################################

def build_missing_mask(
    curve: pd.DataFrame,
) -> pd.DataFrame:
    """
    Возвращает булеву маску пропусков.
    """

    return curve.isna()


def initialize_missing(
    curve: pd.DataFrame,
    method: str = "interpolate",
) -> pd.DataFrame:
    """
    Первоначальная инициализация NaN.

    Parameters
    ----------
    curve

    method

        column_mean

        interpolate

        ffill
    """

    X = curve.copy()

    if method == "column_mean":

        X = X.apply(
            lambda c: c.fillna(
                c.mean()
            )
        )

        return X

    if method == "interpolate":

        X = X.interpolate(
            axis=0,
            limit_direction="both",
        )

        X = X.apply(
            lambda c: c.fillna(
                c.mean()
            )
        )

        return X

    if method == "ffill":

        X = X.ffill()

        X = X.bfill()

        X = X.apply(
            lambda c: c.fillna(
                c.mean()
            )
        )

        return X

    raise ValueError(
        f"Unknown initialization method: {method}"
    )


###############################################################################
# Scaling
###############################################################################

def scale_curve(
    curve: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
):
    """
    Масштабирование матрицы ставок.

    Returns
    -------
    scaled_curve

    fitted_scaler
    """

    if scaler is None:
        scaler = StandardScaler()

        values = scaler.fit_transform(
            curve
        )

    else:

        values = scaler.transform(
            curve
        )

    scaled = pd.DataFrame(
        values,
        index=curve.index,
        columns=curve.columns,
    )

    return scaled, scaler


def inverse_scale_curve(
    curve: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Обратное масштабирование.
    """

    values = scaler.inverse_transform(
        curve
    )

    restored = pd.DataFrame(
        values,
        index=curve.index,
        columns=curve.columns,
    )

    return restored


###############################################################################
# Conversion
###############################################################################

def to_long(
    curve: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wide -> Long
    """

    return (
        curve
        .reset_index()
        .melt(
            id_vars="date",
            var_name="term_bucket",
            value_name="rate",
        )
        .sort_values(
            [
                "date",
                "term_bucket",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def copy_curve(
    curve: pd.DataFrame,
) -> pd.DataFrame:
    """
    Безопасное копирование DataFrame.
    """

    return curve.copy(deep=True)
