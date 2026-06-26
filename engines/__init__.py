"""
yield_curve.engines
===================

Вычислительные движки для восстановления кривых ставок.

Каждый движок реализует единый интерфейс:
    - fit(X, mask)         — обучение на плотной матрице X
                             с булевой маской пропусков
    - transform(X, mask)   — реконструкция новых данных
    - reconstruction_      — атрибут с последней реконструкцией
    - summary()            — диагностическая сводка

Доступные движки
----------------
EMPCAEngine
    Expectation-Maximization PCA. Основной движок,
    устойчив к пропускам, не требует предварительной
    импутации (кроме инициализации).

PPCAEngine (планируется)
    Probabilistic PCA. Даёт вероятностную интерпретацию
    и оценки неопределённости реконструкции.

KalmanEngine (планируется)
    Фильтр Калмана. Учитывает временную структуру
    и даёт сглаженные оценки в реальном времени.

Обычно движки не нужно импортировать напрямую —
интерполятор выбирает нужный по строковому имени:

    interpolator = YieldCurveInterpolator(engine="empca")
"""

from __future__ import annotations

# Готовые движки
from .empca import EMPCAEngine

# Планируемые движки — раскомментировать после реализации:
# from .ppca import PPCAEngine
# from .kalman import KalmanEngine

__all__ = [
    "EMPCAEngine",
    # "PPCAEngine",
    # "KalmanEngine",
]


# Утилитарная функция для проверки валидности имени движка.
# Используется внутри interpolator.py и diagnostics.py.
def available_engines() -> tuple[str, ...]:
    """
    Возвращает кортеж имён всех доступных движков.

    Returns
    -------
    tuple of str
        Например: ('empca',) или ('empca', 'ppca', 'kalman').
    """
    engines = ["empca"]

    # Проверка через try/except, чтобы не падать,
    # если модуль ещё не реализован, но уже импортирован
    try:
        from . import ppca  # noqa: F401
        engines.append("ppca")
    except ImportError:
        pass

    try:
        from . import kalman  # noqa: F401
        engines.append("kalman")
    except ImportError:
        pass

    return tuple(engines)
