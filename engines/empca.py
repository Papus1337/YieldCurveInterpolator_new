"""
yield_curve.engines.empca
=========================

EM-PCA engine.

Алгоритм:

1. Начальная инициализация пропусков
2. PCA
3. Реконструкция
4. Замена только пропущенных значений
5. Relaxation
6. Проверка сходимости
7. Повтор

Данный класс ничего не знает
о DataFrame пользователя.

Он работает исключительно
с числовой матрицей numpy/pandas.
"""

from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


class EMPCAEngine:
    """
    EM-PCA engine.
    """

    def __init__(
        self,
        n_components="auto",
        variance_threshold=0.995,
        max_iter=100,
        tol=1e-5,
        relaxation=0.30,
        random_state=42,
    ):

        self.n_components = n_components

        self.variance_threshold = variance_threshold

        self.max_iter = max_iter

        self.tol = tol

        self.relaxation = relaxation

        self.random_state = random_state

        ##############################################################

        self.pca_: Optional[PCA] = None

        self.components_: Optional[np.ndarray] = None

        self.scores_: Optional[np.ndarray] = None

        self.reconstruction_: Optional[np.ndarray] = None

        ##############################################################

        self.history_: List[Dict] = []

        self.delta_: float = np.inf

        self.n_iter_: int = 0

        ##############################################################

        self.explained_variance_: Optional[np.ndarray] = None

        self.explained_variance_ratio_: Optional[np.ndarray] = None

        self.cumulative_variance_: Optional[np.ndarray] = None

        ##############################################################

        self.is_fitted_: bool = False

    ##################################################################
    # Component selection
    ##################################################################

    def _select_components(
        self,
        X: np.ndarray,
    ) -> int:
        """
        Автоматический выбор числа компонент.
        """

        if self.n_components != "auto":
            return int(self.n_components)

        pca = PCA(
            random_state=self.random_state
        )

        pca.fit(X)

        cumulative = np.cumsum(
            pca.explained_variance_ratio_
        )

        idx = np.searchsorted(
            cumulative,
            self.variance_threshold,
        )

        return max(1, idx + 1)

    ##################################################################
    # PCA fit
    ##################################################################

    def _fit_pca(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Обучение PCA
        и реконструкция данных.
        """

        n_components = self._select_components(X)

        self.pca_ = PCA(
            n_components=n_components,
            random_state=self.random_state,
        )

        scores = self.pca_.fit_transform(X)

        reconstruction = self.pca_.inverse_transform(
            scores
        )

        self.components_ = self.pca_.components_

        self.scores_ = scores

        self.reconstruction_ = reconstruction

        self.explained_variance_ = (
            self.pca_.explained_variance_
        )

        self.explained_variance_ratio_ = (
            self.pca_.explained_variance_ratio_
        )

        self.cumulative_variance_ = np.cumsum(
            self.explained_variance_ratio_
        )

        return reconstruction

    ##################################################################
    # Reconstruction
    ##################################################################

    def _reconstruct(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Реконструкция матрицы через уже обученный PCA.
        """

        scores = self.pca_.transform(X)

        reconstruction = self.pca_.inverse_transform(
            scores
        )

        return reconstruction
    ##################################################################
    # Delta
    ##################################################################

    @staticmethod
    def _compute_delta(
        old: np.ndarray,
        new: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        RMSE только по восстановленным точкам.
        """

        diff = (
            old[mask] -
            new[mask]
        )

        if diff.size == 0:
            return 0.0

        return np.sqrt(
            np.mean(
                diff ** 2
            )
        )

    ##################################################################
    # History
    ##################################################################

    def _update_history(
        self,
        iteration: int,
        delta: float,
    ) -> None:

        self.history_.append(
            {
                "iteration": iteration,
                "delta": delta,
                "components": (
                    self.pca_.n_components_
                    if self.pca_ is not None
                    else None
                ),
                "explained_variance": (
                    self.cumulative_variance_[-1]
                    if self.cumulative_variance_ is not None
                    else None
                ),
            }
        )

    ##################################################################
    # One EM iteration
    ##################################################################

    def _em_iteration(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        fit: bool = True,
    ) -> np.ndarray:
        """
        Одна EM-итерация.

        fit=True  -> переобучить PCA

        fit=False -> использовать уже обученный PCA
        """

        if fit:

            reconstruction = self._fit_pca(X)

        else:

            reconstruction = self._reconstruct(X)

        X_new = X.copy()

        X_new[mask] = (
            self.relaxation * reconstruction[mask]
            + (1.0 - self.relaxation) * X[mask]
        )

        return X_new

    ##################################################################
    # Fit
    ##################################################################

    def fit(
        self,
        X,
        mask,
    ):
        """
        Обучение EM-PCA.

        Parameters
        ----------
        X
            Матрица без NaN.

        mask
            Маска исходных пропусков.
        """

        if isinstance(X, pd.DataFrame):

            columns = X.columns

            index = X.index

            X = X.values

        else:

            columns = None

            index = None

        if isinstance(mask, pd.DataFrame):

            mask = mask.values

        X = X.astype(float)

        current = X.copy()

        self.history_ = []

        self.delta_ = np.inf

        ##############################################################

        for iteration in range(1, self.max_iter + 1):

            updated = self._em_iteration(
                current,
                mask,
            )

            delta = self._compute_delta(
                current,
                updated,
                mask,
            )

            self._update_history(
                iteration,
                delta,
            )

            current = updated

            self.delta_ = delta

            self.n_iter_ = iteration

            if delta < self.tol:
                break

        ##############################################################

        self.reconstruction_ = current

        if columns is not None:

            self.reconstruction_ = pd.DataFrame(
                current,
                index=index,
                columns=columns,
            )

        self.is_fitted_ = True

        return self

    ##################################################################
    # Transform
    ##################################################################

    def transform(
        self,
        X,
        mask,
    ):

        if not self.is_fitted_:

            raise RuntimeError(
                "EMPCAEngine is not fitted."
            )

        dataframe = isinstance(
            X,
            pd.DataFrame,
        )

        if dataframe:

            columns = X.columns

            index = X.index

            current = X.values.astype(float)

        else:

            current = np.asarray(
                X,
                dtype=float,
            )

        if isinstance(mask, pd.DataFrame):

            mask = mask.values

        ##############################################################

        for _ in range(self.max_iter):

            updated = self._em_iteration(
                current,
                mask,
                fit=False,
            )

            delta = self._compute_delta(
                current,
                updated,
                mask,
            )

            current = updated

            if delta < self.tol:

                break

        ##############################################################

        if dataframe:

            return pd.DataFrame(
                current,
                index=index,
                columns=columns,
            )

        return current

    ##################################################################
    # Fit transform
    ##################################################################

    def fit_transform(
        self,
        X,
        mask,
    ):

        self.fit(
            X,
            mask,
        )

        return self.reconstruction_

    ##################################################################
    # Summary
    ##################################################################

    def summary(
        self,
    ) -> Dict:

        if not self.is_fitted_:

            raise RuntimeError(
                "EMPCAEngine is not fitted."
            )

        return {

            "iterations": self.n_iter_,

            "final_delta": self.delta_,

            "n_components": (
                self.pca_.n_components_
                if self.pca_ is not None
                else None
            ),

            "explained_variance": (
                float(
                    self.cumulative_variance_[-1]
                )
                if self.cumulative_variance_ is not None
                else None
            ),

            "converged": (
                self.delta_ < self.tol
            ),

        }

    ##################################################################
    # Reset
    ##################################################################

    def reset(
        self,
    ):

        self.__init__(
            n_components=self.n_components,
            variance_threshold=self.variance_threshold,
            max_iter=self.max_iter,
            tol=self.tol,
            relaxation=self.relaxation,
            random_state=self.random_state,
        )

        return self

    ##################################################################
    # Representation
    ##################################################################

    def __repr__(
        self,
    ):

        status = (
            "fitted"
            if self.is_fitted_
            else "not fitted"
        )

        return (
            f"EMPCAEngine("
            f"status={status}, "
            f"iterations={self.n_iter_}, "
            f"delta={self.delta_:.3e})"
        )
