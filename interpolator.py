from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import pandas as pd

from . import preprocessing as pp
from .engines.empca import EMPCAEngine

# Позже можно будет добавить импорты других движков:
# from .engines.ppca import PPCAEngine
# from .engines.kalman import KalmanEngine


class YieldCurveInterpolator:
    """
    Публичный API (Facade) для моделирования рыночных кривых ставок.
    
    Класс объединяет все этапы конвейера:
    1. Валидацию и подготовку данных (PreProcessing).
    2. Масштабирование признаков.
    3. Работу с вычислительным движком (EM-PCA, PPCA, Kalman).
    4. Обратное масштабирование и формирование итогового отклика.
    
    Параметры
    ----------
    engine : str
        Название движка для интерполяции ('empca', 'ppca', 'kalman').
    engine_params : dict, optional
        Словарь параметров для инициализации движка.
    buckets : Iterable[int], optional
        Допустимые бакеты срочности.
    init_method : str, optional
        Метод первичной инициализации пропусков до подачи в движок.
    scale : bool, optional
        Флаг необходимости масштабирования матрицы ставок.
    """

    def __init__(
        self,
        engine: str = "empca",
        engine_params: Optional[Dict[str, Any]] = None,
        buckets: Iterable[int] = pp.DEFAULT_BUCKETS,
        init_method: str = "interpolate",
        scale: bool = True,
    ):
        self.engine_name = engine
        self.engine_params = engine_params or {}
        self.buckets = tuple(buckets)
        self.init_method = init_method
        self.scale = scale
        
        self._engine = self._init_engine()
        self._scaler: Optional[Any] = None
        
        # Состояние модели
        self.curve_: Optional[pd.DataFrame] = None
        self.mask_: Optional[pd.DataFrame] = None
        self.original_curve_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False
        
        # Сохраняем средние из истории для инициализации
        self._columns_means_: Optional[pd.Series] = None

    def _init_engine(self) -> Any:
        """Фабричный метод для создания вычислительного движка."""
        if self.engine_name == "empca":
            return EMPCAEngine(**self.engine_params)
            
        # Заготовки для будущих движков
        # elif self.engine_name == "ppca":
        #     return PPCAEngine(**self.engine_params)
        # elif self.engine_name == "kalman":
        #     return KalmanEngine(**self.engine_params)
        
        raise ValueError(f"Неизвестный движок интерполяции: {self.engine_name}")

    def _preprocess(
        self, 
        df: pd.DataFrame, 
        is_fit: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Общий контур предобработки.
        Возвращает: (подготовленная матрица, маска пропусков, исходная матрица)
        """
        pp.validate_input(df, self.buckets)
        
        # Формируем широкую матрицу
        original_curve = pp.build_rate_matrix(df, self.buckets)
        mask = pp.build_missing_mask(original_curve)
        
        ## Инициализируем пропуски, чтобы матрица была плотной (требование движков)
        #initialized = pp.initialize_missing(original_curve, method=self.init_method)
        
        # Инициализируем пропуски
        if is_fit:
            # При обучении используем стандартный метод
            initialized = pp.initialize_missing(original_curve, method=self.init_method)
            # Сохраняем средние по колонкам для будущего использования
            self._columns_means_ = initialized.mean()
        else:
            # При трансформации: если строк < 2, то используем средние из истории
            if len(original_curve) < 2:
                initialized = original_curve.copy()
                # Заполняем NaN средними из истории
                for col in initialized.columns:
                    if initialized[col].isna().any() and self._columns_means_ is not None:
                        initialized[col] = initialized[col].fillna(self._columns_means_[col])
                # Если всё ещё есть NaN, заполняем нулями
                initialized = initialized.fillna(0.0)
            else:
                initialized = pp.initialize_missing(original_curve, method=self.init_method)
        
        # Масштабирование
        if self.scale:
            if is_fit:
                scaled, self._scaler = pp.scale_curve(initialized)
            else:
                if self._scaler is None:
                    raise RuntimeError(
                        "Масштабировщик не обучен. Сначала вызовите fit()."
                    )
                # При трансформации используем уже обученный скейлер
                scaled, _ = pp.scale_curve(initialized, scaler=self._scaler)
        else:
            scaled = initialized
            
        return scaled, mask, original_curve

    def _postprocess(
        self, 
        reconstructed: pd.DataFrame, 
        original_curve: pd.DataFrame
    ) -> pd.DataFrame:
        """Обратное масштабирование и восстановление структуры."""
        if self.scale and self._scaler is not None:
            restored = pp.inverse_scale_curve(reconstructed, self._scaler)
        else:
            restored = reconstructed.copy()
            
        # Гарантируем сохранение исходной разметки (индексов и колонок)
        restored.index = original_curve.index
        restored.columns = original_curve.columns
        
        return restored

    def fit(self, df: pd.DataFrame) -> "YieldCurveInterpolator":
        """
        Обучение движка на исторических данных.
        
        Parameters
        ----------
        df : pd.DataFrame
            Исходные данные в длинном формате (date, term_bucket, rate, amount).
        """
        scaled, mask, original_curve = self._preprocess(df, is_fit=True)
        
        # Передаем плотную матрицу и маску в движок
        self._engine.fit(scaled, mask)
        
        # Получаем реконструкцию из движка (после fit reconstruction_ хранит результат)
        reconstructed_scaled = self._engine.reconstruction_
        
        # Обратная трансформация и сохранение состояния
        self.curve_ = self._postprocess(reconstructed_scaled, original_curve)
        self.mask_ = mask
        self.original_curve_ = original_curve
        self.is_fitted_ = True
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Интерполяция новых данных с использованием обученного движка.
        
        Parameters
        ----------
        df : pd.DataFrame
            Новые данные в длинном формате.
            
        Returns
        -------
        pd.DataFrame
            Интерполированная кривая ставок (широкий формат).
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Интерполятор не обучен. Сначала вызовите fit()."
            )
            
        scaled, mask, original_curve = self._preprocess(df, is_fit=False)
        
        # Трансформация без переобучения PCA (fit=False внутри transform у EMPCA)
        reconstructed_scaled = self._engine.transform(scaled, mask)
        
        return self._postprocess(reconstructed_scaled, original_curve)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обучение и немедленная интерполяция тех же данных.
        """
        self.fit(df)
        return self.curve_

    def get_curve(self) -> pd.DataFrame:
        """
        Получение последней интерполированной кривой.
        """
        if self.curve_ is None:
            raise RuntimeError(
                "Кривая не найдена. Сначала вызовите fit() или transform()."
            )
        return self.curve_

    def get_summary(self) -> Dict[str, Any]:
        """
        Получение диагностической информации о сходимости движка.
        """
        if not self.is_fitted_:
            raise RuntimeError("Интерполятор не обучен.")
            
        return self._engine.summary()

    def get_components(self) -> Optional[pd.DataFrame]:
        """
        Получение компонент (факторов) кривой, если движок их предоставляет.
        Полезно для анализа того, какие факторы (уровень, наклон, выпуклость) 
        захватывает модель.
        """
        if hasattr(self._engine, "components_") and self._engine.components_ is not None:
            columns = self.curve_.columns if self.curve_ is not None else None
            return pd.DataFrame(
                self._engine.components_,
                columns=columns,
            )
        return None

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"YieldCurveInterpolator("
            f"engine='{self.engine_name}', "
            f"status={status})"
        )
