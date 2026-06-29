# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:15:10 2026

@author: mb.aliev
"""

import warnings
warnings.filterwarnings("ignore")

# Standard python libraries
import os
import re
import datetime
import time

# Installed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyodbc 

def DB_request(sql):
    
    server = 'trading-db.ahml1.ru'
    database = 'LIQUIDITY'
    
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';' \
                                          'DATABASE=' + database + ';Trusted_connection=yes')
    cursor = conn.cursor()

    t1 = time.time()
    
    try:
        df = pd.read_sql(sql, conn)
        print('Request completed: ', time.time() - t1)
    except Exception as e:
        print('Request execution error:', e)
        
    cursor.close()
    conn.close()
    
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    
    return df

# -------------------------------------------------------
# 1. Загрузка реальных данных
# -------------------------------------------------------
# Предполагаем, что у тебя есть CSV/Excel файлы
# Формат: date, term_bucket, rate, amount

df1 = """
        select [date], TERM_AVG term_bucket, spread rate, amount
        from [LIQUIDITY].[qt].prev_spreads_from_auc_V2_2_WithPERC 
        where 1 = 1
        order by [date], term_avg
"""
df1 = DB_request(df1)
df2 = """
        select [date], TERM_AVG term_bucket, spread rate
        from [LIQUIDITY].[qt].[spreads_from_auc_V2] 
        where 1 = 1
        and [date] > '2026-06-01'
        order by [date], term_avg
"""
df2 = DB_request(df2)


df_train = df1[df1['date'] < '2026-06-01']
df_eval = df1[df1['date'] >= '2026-06-01']
df_benchmark = df2

# -------------------------------------------------------
# 2. Проверка и очистка
# -------------------------------------------------------
from yield_curve import preprocessing as pp

# Бакеты, которые будем моделировать
BUCKETS = [1, 7, 14, 31, 61, 91, 181]

# Валидация каждого датафрейма
for name, df in [("train", df_train), ("eval", df_eval)]:
    print(f"\n=== Проверка {name} ===")
    print(f"Размер: {df.shape}")
    print(f"Период: {df['date'].min()} — {df['date'].max()}")
    print(f"Пропусков в rate: {df['rate'].isna().sum()}")
    print(f"Пропусков в amount: {df['amount'].isna().sum()}")
    
    pp.validate_input(df, buckets=BUCKETS)
    print("Валидация пройдена.")

# -------------------------------------------------------
# 3. Подготовка бенчмарка (wide-формат)
# -------------------------------------------------------
# Бенчмарк может быть уже в wide-формате (index=date, columns=buckets)
# Если он в long-формате — конвертируем:

if set(df_benchmark.columns) >= {"date", "term_bucket", "rate"}:
    # Если в бенчмарке нет amount — добавим единичные веса
    if "amount" not in df_benchmark.columns:
        df_benchmark["amount"] = 1.0
    
    # Используем build_rate_matrix для единообразия
    benchmark_curve = pp.build_rate_matrix(df_benchmark, buckets=BUCKETS)
elif set(df_benchmark.columns) >= set(BUCKETS):
    # Уже wide-формат
    benchmark_curve = df_benchmark.set_index("date")
    benchmark_curve.columns = [int(c) for c in benchmark_curve.columns]
    benchmark_curve = benchmark_curve.reindex(columns=BUCKETS)
else:
    raise ValueError("Неизвестный формат бенчмарка")

print(f"\nБенчмарк: {benchmark_curve.shape}")
print(f"Период: {benchmark_curve.index.min()} — {benchmark_curve.index.max()}")


from yield_curve import YieldCurveInterpolator

# -------------------------------------------------------
# 4. Обучение интерполятора
# -------------------------------------------------------
interpolator = YieldCurveInterpolator(
    engine="empca",
    engine_params={
        "n_components": 3,
        "max_iter": 200,
        #"tol": 1e-7,
        "relaxation": 0.8,
    },
    buckets=BUCKETS,
    init_method="interpolate",
    scale=True,
)

print("\n=== Обучение модели ===")
interpolator.fit(df_train)

# Проверка сходимости
from yield_curve import diagnostics
summary = interpolator._engine.summary()
conv = diagnostics.check_convergence(summary)
print(f"Сходимость: {conv['verdict']}")
print(f"Итераций: {conv['n_iter']}")
print(f"Финальная ошибка: {conv['final_error']:.2e}")

if conv["verdict"] == "FAIL":
    print("ВНИМАНИЕ: модель не сошлась. Увеличьте max_iter или ослабьте tolerance.")

# Обученная кривая
train_curve = interpolator.get_curve()
print(f"\nКривая на train: {train_curve.shape}")
print(f"Пропусков: {train_curve.isna().sum().sum()}")


# Используем модель, обученную только на train
# Трансформируем eval-данные через уже обученную модель
eval_curve = interpolator.transform(df_eval)


# -------------------------------------------------------
# 5. Выравнивание кривых
# -------------------------------------------------------

# Общее пересечение дат
common_dates = eval_curve.index.intersection(benchmark_curve.index)
print(f"\nОбщих дат: {len(common_dates)}")
print(f"Период сравнения: {common_dates.min()} — {common_dates.max()}")

if len(common_dates) == 0:
    raise ValueError(
        "Нет пересечения дат между предсказанной кривой и бенчмарком. "
        "Проверьте, что бенчмарк покрывает период eval."
    )

# Фильтруем обе кривые
predicted_aligned = eval_curve.loc[common_dates]
actual_aligned = benchmark_curve.loc[common_dates]

# Проверка на пропуски в бенчмарке
missing_in_benchmark = actual_aligned.isna().sum().sum()
if missing_in_benchmark > 0:
    print(f"В бенчмарке {missing_in_benchmark} пропусков. Они будут исключены из сравнения.")

# Маска пропусков из интерполятора для eval-периода строится отдельно,
# потому что interpolate.mask_ хранит маску только для train
eval_rate_matrix = pp.build_rate_matrix(df_eval, buckets=BUCKETS)
eval_mask = pp.build_missing_mask(eval_rate_matrix)

# выравниваем маску с common_dates
mask_aligned = eval_mask.loc[common_dates] if eval_mask is not None else None


from yield_curve import metrics

# -------------------------------------------------------
# 6. Сравнение с бенчмарком
# -------------------------------------------------------
report = metrics.compare_curves(
    predicted=predicted_aligned,
    actual=actual_aligned,
    mask=mask_aligned,
    notional=10_000_000,  # 10 млн на бакет
)

metrics.print_metrics_report(report)

# -------------------------------------------------------
# 7. Детализация по бакетам
# -------------------------------------------------------
print("\n=== Метрики по бакетам ===")
print(report["metrics_by_bucket"].to_string())

# -------------------------------------------------------
# 8. Детализация по датам
# -------------------------------------------------------
print("\n=== Худшие 10 дат по RMSE ===")
worst_dates = report["metrics_by_date"].sort_values("rmse", ascending=False).head(10)
print(worst_dates.to_string())

# -------------------------------------------------------
# 9. Направленная точность
# -------------------------------------------------------
print(f"\nНаправленная точность: {report['directional_accuracy']:.4f}")
print(f"Ранговая корреляция (Spearman): {report['rank_correlation'].mean():.4f}")

# -------------------------------------------------------
# 10. Экономические метрики
# -------------------------------------------------------
print(f"\nСуммарный PnL: {report['total_pnl']:,.2f}")
print(f"Годовой PnL: {report['annualized_pnl']:,.2f}")


from yield_curve import plotting

# -------------------------------------------------------
# 11. Визуализация
# -------------------------------------------------------

# Сравнение на конкретную дату
sample_date = common_dates[len(common_dates) // 2]
fig_compare = plotting.plot_curves_comparison(
    observed=actual_aligned,
    fitted=predicted_aligned,
    date=sample_date,
)
fig_compare.show()

# Динамика ставок по бакетам
fig_ts = plotting.plot_curve_timeseries(predicted_aligned)
fig_ts.show()

# Heatmap остатков
residuals = actual_aligned - predicted_aligned
fig_heatmap = plotting.plot_residuals_heatmap(residuals)
fig_heatmap.show()

# PnL
pnl_df = metrics.pnl_per_bucket(predicted_aligned, actual_aligned)
fig_pnl = plotting.plot_pnl_timeseries(pnl_df)
fig_pnl.show()

# -------------------------------------------------------
# 12. Экспорт для шаринга
# -------------------------------------------------------
plotting.save_figure(fig_compare, "reports/curve_comparison.html", format="html")
plotting.save_figure(fig_heatmap, "reports/residuals_heatmap.html", format="html")
plotting.save_figure(fig_pnl, "reports/pnl_curve.html", format="html")

# Если нужен PNG для вставки в Outlook
plotting.save_figure(fig_compare, "reports/curve_comparison.png", format="png", scale=2)

print("\nОтчёты сохранены в директорию reports/")


from yield_curve.backtest import CurveBacktester, plot_backtest_dashboard

# -------------------------------------------------------
# 13. Бэктест mean-reversion стратегии
# -------------------------------------------------------
backtester = CurveBacktester(
    strategy="mean_reversion",
    strategy_params={
        "zscore_threshold": 1.0,
        "lookback": 20,
    },
    notional=10_000_000,
    transaction_cost_bps=5.0,
    slippage_bps=2.0,
    signal_lag=1,
    rebalance_frequency=1,
)

result = backtester.run(
    predicted=predicted_aligned,
    actual=actual_aligned,
)

result.print_summary()

# Визуализация бэктеста
fig_bt = plot_backtest_dashboard(result)
fig_bt.show()
plotting.save_figure(fig_bt, "reports/backtest_dashboard.html", format="html")

# -------------------------------------------------------
# 14. Сравнение стратегий
# -------------------------------------------------------
from yield_curve.backtest import compare_strategies

comparison = compare_strategies(
    predicted=predicted_aligned,
    actual=actual_aligned,
    strategies=["directional", "mean_reversion", "carry"],
    notional=10_000_000,
    transaction_cost_bps=5.0,
)

print("\n=== Сравнение стратегий ===")
print(comparison.to_string())

# Сохранение таблицы
comparison.to_csv("reports/strategy_comparison.csv")


runfile('C:/Users/mb.aliev/Desktop/PY_apps/!projects/prod_test.py', wdir='C:/Users/mb.aliev/Desktop/PY_apps/!projects')
Reloaded modules: yield_curve.preprocessing, yield_curve.engines.empca, yield_curve.engines, yield_curve.interpolator, yield_curve.diagnostics, yield_curve.metrics, yield_curve
Request completed:  0.056951284408569336
Request completed:  3.980391502380371

=== Проверка train ===
Размер: (7161, 4)
Период: 2022-04-08 00:00:00 — 2026-05-29 00:00:00
Пропусков в rate: 3691
Пропусков в amount: 3691
Валидация пройдена.

=== Проверка eval ===
Размер: (140, 4)
Период: 2026-06-01 00:00:00 — 2026-06-29 00:00:00
Пропусков в rate: 73
Пропусков в amount: 73
Валидация пройдена.

Бенчмарк: (19, 7)
Период: 2026-06-02 00:00:00 — 2026-06-29 00:00:00

=== Обучение модели ===
Сходимость: FAIL
Итераций: 0
Финальная ошибка: inf
ВНИМАНИЕ: модель не сошлась. Увеличьте max_iter или ослабьте tolerance.

Кривая на train: (1023, 7)
Пропусков: 0

Общих дат: 19
Период сравнения: 2026-06-02 00:00:00 — 2026-06-29 00:00:00
Traceback (most recent call last):

  File C:\ProgramData\anaconda3\lib\site-packages\spyder_kernels\py3compat.py:356 in compat_exec
    exec(code, globals, locals)

  File c:\users\mb.aliev\desktop\py_apps\!projects\prod_test.py:200
    report = metrics.compare_curves(

  File ~\Desktop\PY_apps\!projects\yield_curve\metrics.py:726 in compare_curves
    report["pointwise"] = pointwise_metrics(p, a, aligned_mask)

  File ~\Desktop\PY_apps\!projects\yield_curve\metrics.py:228 in pointwise_metrics
    "rmse": rmse(predicted, actual, mask),

  File ~\Desktop\PY_apps\!projects\yield_curve\metrics.py:131 in rmse
    p, a = _align_curves(predicted, actual, mask)

ValueError: too many values to unpack (expected 2)
