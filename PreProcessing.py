"""
yield_curve.preprocessing
=========================

Модуль подготовки данных для моделей восстановления
рыночной кривой депозитных ставок.

Функциональность
----------------
- Проверка входных данных
- Построение матрицы ставок
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
    Построение матрицы ставок.

    Returns
    -------
    DataFrame

        index   -> date

        columns -> maturity

        values  -> weighted rate
    """

    curve = (
        df
        .pivot_table(
            index="date",
            columns="term_bucket",
            values="rate",
            aggfunc="mean",
        )
        .sort_index()
    )

    curve = curve.reindex(columns=buckets)

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
