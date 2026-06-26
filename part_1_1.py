# =============================================================================
# empca_interpolator.py
#
# EM-PCA Yield Curve Interpolator
#
# Автор:
#
# Версия: 0.1
#
# =============================================================================

from __future__ import annotations

import warnings

from typing import Optional
from typing import Dict
from typing import List
from typing import Any

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class EMPCAInterpolator:
    """
    ===========================================================================
    EM-PCA Yield Curve Interpolator
    ===========================================================================

    Назначение
    ----------
    Восстановление пропущенных точек рыночной депозитной кривой
    посредством EM-PCA.

    Ожидаемый формат входных данных

        date
        term_bucket
        rate
        amount

    На выходе

        date
        term_bucket
        rate

    Архитектура

        Long dataframe

            ↓

        Pivot

            ↓

        Initial Imputation

            ↓

        Standardization

            ↓

        EM-PCA

            ↓

        Inverse Standardization

            ↓

        Long dataframe

    """

    ###########################################################################
    # CONSTANTS
    ###########################################################################

    BUCKETS = [
        1,
        7,
        14,
        31,
        61,
        91,
        181
    ]

    REQUIRED_COLUMNS = {
        "date",
        "term_bucket",
        "rate",
        "amount"
    }

    ###########################################################################
    # Constructor
    ###########################################################################

    def __init__(
        self,
        n_components="auto",
        variance_threshold=0.995,
        max_iter=100,
        tol=1e-5,
        relaxation=0.30,
        scale=True,
        init_method="interpolate",
        verbose=False,
        random_state=42,
    ):

        #######################################################################
        # User parameters
        #######################################################################

        self.n_components = n_components

        self.variance_threshold = variance_threshold

        self.max_iter = max_iter

        self.tol = tol

        self.relaxation = relaxation

        self.scale = scale

        self.init_method = init_method

        self.verbose = verbose

        self.random_state = random_state

        #######################################################################
        # Models
        #######################################################################

        self.scaler = None

        self.pca = None

        #######################################################################
        # Metadata
        #######################################################################

        self.index_ = None

        self.columns_ = None

        #######################################################################
        # Matrices
        #######################################################################

        self.rate_matrix_ = None

        self.amount_matrix_ = None

        self.original_curve_ = None

        self.initial_curve_ = None

        self.scaled_curve_ = None

        self.reconstructed_curve_ = None

        self.final_curve_ = None

        #######################################################################
        # Masks
        #######################################################################

        self.missing_mask_ = None

        #######################################################################
        # PCA diagnostics
        #######################################################################

        self.components_ = None

        self.scores_ = None

        self.loadings_ = None

        self.explained_variance_ = None

        self.explained_variance_ratio_ = None

        self.cumulative_variance_ = None

        #######################################################################
        # EM diagnostics
        #######################################################################

        self.history_ = []

        self.n_iter_ = 0

        self.delta_ = np.inf

        #######################################################################
        # Status
        #######################################################################

        self.is_fitted_ = False

    ###########################################################################
    # VALIDATION
    ###########################################################################

    def _validate_input(
        self,
        df: pd.DataFrame
    ) -> None:

        """
        Проверка корректности входного датафрейма.
        """

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "Input must be pandas.DataFrame"
            )

        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if len(missing):

            raise ValueError(
                f"Missing columns: {missing}"
            )

        maturities = sorted(df.term_bucket.unique())

        unknown = set(maturities) - set(self.BUCKETS)

        if len(unknown):

            raise ValueError(
                f"Unknown term buckets: {unknown}"
            )

        if df.empty:

            raise ValueError(
                "Input dataframe is empty."
            )

    ###########################################################################
    # RATE MATRIX
    ###########################################################################

    def _build_rate_matrix(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Строит матрицу

                date × maturity

        из ставок.
        """

        curve = (
            df
            .pivot_table(
                index="date",
                columns="term_bucket",
                values="rate",
                aggfunc="mean"
            )
            .sort_index()
        )

        curve = curve.reindex(
            columns=self.BUCKETS
        )

        curve.index = pd.to_datetime(
            curve.index
        )

        return curve

    ###########################################################################
    # AMOUNT MATRIX
    ###########################################################################

    def _build_amount_matrix(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Строит аналогичную матрицу объемов.

        Пока используется только
        для диагностики.

        Позже будет использоваться
        в Weighted PCA.
        """

        amount = (
            df
            .pivot_table(
                index="date",
                columns="term_bucket",
                values="amount",
                aggfunc="sum"
            )
            .sort_index()
        )

        amount = amount.reindex(
            columns=self.BUCKETS
        )

        amount.index = pd.to_datetime(
            amount.index
        )

        return amount

    ###########################################################################
    # LONG FORMAT
    ###########################################################################

    def _to_long(
        self,
        curve: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Преобразование обратно
        в long dataframe.
        """

        out = (
            curve
            .reset_index()
            .melt(
                id_vars="date",
                var_name="term_bucket",
                value_name="rate"
            )
            .sort_values(
                [
                    "date",
                    "term_bucket"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return out
