"""
yield_curve
===========

Библиотека для моделирования рыночных кривых депозитных ставок
на основе методов снижения размерности (EM-PCA, PPCA, Kalman).

Основной публичный API
----------------------
YieldCurveInterpolator
    Фасад, объединяющий предобработку, масштабирование,
    вычислительный движок и обратную трансформацию.

Подмодули
---------
preprocessing
    Подготовка данных: валидация, построение матриц,
    взвешенная агрегация по объёму, инициализация пропусков,
    масштабирование.

diagnostics
    Внутренняя диагностика модели: сходимость, объяснённая
    дисперсия, анализ остатков, поиск выбросов.

metrics
    Внешние метрики качества: RMSE, MAE, directional accuracy,
    косинусное сходство формы, экономические метрики (PnL).

plotting
    Визуализация кривых, компонентов, остатков, сходимости.

backtest
    Бэктест торговых стратегий на сигналах модели.

engines
    Вычислительные движки (EMPCA, PPCA, Kalman).
    Обычно не требуется импортировать напрямую —
    интерполятор выбирает движок по строковому имени.

utils
    Вспомогательные функции (внутреннее использование).
"""

from __future__ import annotations

# Версия библиотеки — единственная точка правды
__version__ = "0.1.0"

# Главный публичный класс — то, с чего начинает работу 99% пользователей
from .interpolator import YieldCurveInterpolator

# Подмодули как namespace.
# Это позволяет делать:
#   from yield_curve import metrics
#   metrics.rmse(predicted, actual)
# или:
#   import yield_curve
#   yield_curve.diagnostics.full_report(interpolator)
#
# Без этих строк пришлось бы писать:
#   from yield_curve import metrics  # работает и так, но явно лучше
#
# Явный импорт подмодулей гарантирует, что они будут доступны
# как атрибуты пакета после `import yield_curve`.
from . import preprocessing
from . import diagnostics
from . import metrics
from . import engines

# plotting и backtest будут добавлены, когда модули будут готовы.
# from . import plotting
# from . import backtest

# utils — внутренний модуль, не экспортируем в публичный API.
# Доступен как yield_curve.utils для продвинутых пользователей.

__all__ = [
    # Версия
    "__version__",
    # Главный класс
    "YieldCurveInterpolator",
    # Подмодули (namespace)
    "preprocessing",
    "diagnostics",
    "metrics",
    "engines",
]
