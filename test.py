import numpy as np
import pandas as pd
from yield_curve import YieldCurveInterpolator
from yield_curve import diagnostics, metrics, plotting
from yield_curve.backtest import CurveBacktester

# -------------------------------------------------------
# 1. Генерация синтетических данных для демонстрации
# -------------------------------------------------------
np.random.seed(42)

dates = pd.date_range("2024-01-01", periods=200, freq="B")  # рабочие дни
buckets = [1, 7, 14, 31, 61, 91, 181]

# Базовая кривая: уровень ~8%, наклон ~2%, выпуклость ~0.5%
records = []
for date in dates:
    level = 8.0 + np.cumsum(np.random.randn(1) * 0.05)[-1]
    slope = 2.0 + np.random.randn() * 0.1
    curvature = 0.5 + np.random.randn() * 0.05

    for bucket in buckets:
        # Факторная модель: rate = level + slope * f1(bucket) + curvature * f2(bucket) + noise
        f1 = np.log(1 + bucket) / np.log(1 + 181)  # нормализованный наклон
        f2 = np.sin(np.pi * bucket / 181)          # выпуклость
        rate = level + slope * f1 + curvature * f2 + np.random.randn() * 0.05

        # Объём: зависит от бакета и даты
        amount = np.random.lognormal(mean=15, sigma=1.0)

        # Пропуски: ~10% данных отсутствуют
        if np.random.rand() < 0.1:
            rate = np.nan

        records.append({
            "date": date,
            "term_bucket": bucket,
            "rate": rate,
            "amount": amount,
        })

df = pd.DataFrame(records)

# -------------------------------------------------------
# 2. Обучение интерполятора
# -------------------------------------------------------
interpolator = YieldCurveInterpolator(
    engine="empca",
    engine_params={"n_components": 3, "max_iter": 100, "tolerance": 1e-6},
    buckets=buckets,
    init_method="interpolate",
    scale=True,
)

interpolator.fit(df)
curve = interpolator.get_curve()

print(f"Кривая восстановлена: {curve.shape[0]} дат x {curve.shape[1]} бакетов")
print(f"Пропусков в исходных данных: {df['rate'].isna().sum()}")
print(f"Пропусков после интерполяции: {curve.isna().sum().sum()}")

# -------------------------------------------------------
# 3. Диагностика модели
# -------------------------------------------------------
report = diagnostics.full_report(interpolator)
diagnostics.print_report(report)

# -------------------------------------------------------
# 4. Визуализация
# -------------------------------------------------------
# Снимок кривой на последнюю дату
fig_snapshot = plotting.plot_curve_snapshot(curve, date=curve.index[-1])
fig_snapshot.show()

# Сводный дашборд
fig_dashboard = plotting.plot_full_dashboard(interpolator)
fig_dashboard.show()

# Сохранение в HTML для шаринга через сеть
plotting.save_figure(fig_dashboard, "dashboard.html", format="html")

# -------------------------------------------------------
# 5. Сравнение с эталоном (если есть)
# -------------------------------------------------------
# В реальном сценарии actual — это официальная кривая ЦБ
# Для демонстрации добавим шум к реконструированной кривой
actual = curve + np.random.randn(*curve.shape) * 0.02
actual = pd.DataFrame(actual, index=curve.index, columns=curve.columns)

metrics_report = metrics.compare_curves(curve, actual)
metrics.print_metrics_report(metrics_report)

# -------------------------------------------------------
# 6. Бэктест торговой стратегии
# -------------------------------------------------------
backtester = CurveBacktester(
    strategy="mean_reversion",
    strategy_params={"zscore_threshold": 0.5, "lookback": 20},
    notional=1_000_000,
    transaction_cost_bps=5.0,
    slippage_bps=2.0,
    signal_lag=1,
)

result = backtester.run(predicted=curve, actual=actual)
result.print_summary()

# Визуализация бэктеста
from yield_curve.backtest import plot_backtest_dashboard
fig_bt = plot_backtest_dashboard(result)
fig_bt.show()
plotting.save_figure(fig_bt, "backtest.html", format="html")
