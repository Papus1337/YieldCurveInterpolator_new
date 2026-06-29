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
        "max_iter": 500,
        "tol": 1e-6,
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
Reloaded modules: yield_curve.preprocessing, yield_curve.engines.empca, yield_curve.engines, yield_curve.interpolator, yield_curve.diagnostics, yield_curve.metrics, yield_curve, yield_curve.plotting
Request completed:  0.2095036506652832
Request completed:  0.2111506462097168

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
Итераций: 1000
Финальная ошибка: 1.17e-04
ВНИМАНИЕ: модель не сошлась. Увеличьте max_iter или ослабьте tolerance.

Кривая на train: (1023, 7)
Пропусков: 0

Общих дат: 19
Период сравнения: 2026-06-02 00:00:00 — 2026-06-29 00:00:00
======================================================================
YIELD CURVE METRICS REPORT
======================================================================

Total observations: 133

[1] POINTWISE METRICS
    rmse        : 0.000417
    mae         : 0.000239
    mape        : 49.1060%
    max_error   : 0.002203

[2] DIRECTIONAL ACCURACY
    0.9302 (93.02%)

[3] RANK CORRELATION (Spearman)
    Mean: 0.8410
    Min:  0.2000
    Max:  1.0000

[4] ECONOMIC METRICS
    Total PnL:                -21.69
    Annualized PnL:          -287.68

[5] METRICS BY BUCKET (top 3 worst by RMSE)
            rmse       mae        mape  max_error  directional_accuracy  mean_bias  n_obs
bucket                                                                                   
61      0.001065  0.000766  208.722647   0.002203                  0.75  -0.000674      5
91      0.000603  0.000361  188.455624   0.001316                  0.75   0.000246      5
31      0.000302  0.000232   39.787272   0.000546                  1.00   0.000032      9

[6] METRICS BY DATE (top 5 worst by RMSE)
                rmse       mae  max_error  l2_distance  cosine_similarity  n_obs
date                                                                            
2026-06-26  0.001147  0.000817   0.002203     0.002294           0.727355      4
2026-06-25  0.000609  0.000439   0.001316     0.001491           0.939998      6
2026-06-22  0.000512  0.000387   0.000744     0.001023           0.976855      4
2026-06-17  0.000469  0.000441   0.000612     0.000812           0.990642      3
2026-06-23  0.000429  0.000313   0.000728     0.000744           0.995184      3

======================================================================

=== Метрики по бакетам ===
            rmse       mae        mape  max_error  directional_accuracy  mean_bias  n_obs
bucket                                                                                   
1       0.000251  0.000179    4.520187   0.000698              0.875000  -0.000066     17
7       0.000285  0.000195   14.180142   0.000728              0.923077   0.000012     14
14      0.000193  0.000130   39.484332   0.000496              1.000000   0.000042     12
31      0.000302  0.000232   39.787272   0.000546              1.000000   0.000032      9
61      0.001065  0.000766  208.722647   0.002203              0.750000  -0.000674      5
91      0.000603  0.000361  188.455624   0.001316              0.750000   0.000246      5
181     0.000010  0.000010    0.522118   0.000010                   NaN  -0.000010      1

=== Худшие 10 дат по RMSE ===
                rmse       mae  max_error  l2_distance  cosine_similarity  n_obs
date                                                                            
2026-06-26  0.001147  0.000817   0.002203     0.002294           0.727355      4
2026-06-25  0.000609  0.000439   0.001316     0.001491           0.939998      6
2026-06-22  0.000512  0.000387   0.000744     0.001023           0.976855      4
2026-06-17  0.000469  0.000441   0.000612     0.000812           0.990642      3
2026-06-23  0.000429  0.000313   0.000728     0.000744           0.995184      3
2026-06-19  0.000336  0.000275   0.000546     0.000581           0.995865      3
2026-06-15  0.000222  0.000167   0.000313     0.000314           0.999111      2
2026-06-03  0.000216  0.000153   0.000367     0.000373           0.999655      3
2026-06-18  0.000187  0.000178   0.000234     0.000264           0.999527      2
2026-06-10  0.000182  0.000140   0.000311     0.000445           0.997975      6

Направленная точность: 0.9302
Ранговая корреляция (Spearman): 0.8410

Суммарный PnL: -21.69
Годовой PnL: -287.68





Отчёты сохранены в директорию reports/
======================================================================
BACKTEST REPORT — MEAN_REVERSION
======================================================================

[Period]
    Start:              2026-06-02 00:00:00
    End:                2026-06-29 00:00:00
    Trading days:       19

[Returns]
    Total return:           -335,909.02
    Total return (%):           -0.4799%
    Annualized return:          -0.0651 (-6.51%)
    Annualized vol:              0.0029 (0.29%)

[Risk-adjusted]
    Sharpe ratio:              -22.2125
    Sortino ratio:             -24.5175
    Calmar ratio:              -13.5716
    Max drawdown:               -0.0048 (-0.48%)
    Max DD duration:                 16 days

[Trading]
    Win rate:                    0.0000 (0.00%)
    Profit factor:               0.0000
    Number of trades:                44
    Avg daily turnover:   25,263,157.89

[Parameters]
    notional              : 10000000
    transaction_cost_bps  : 5.0
    slippage_bps          : 2.0
    risk_free_rate        : 0.0
    signal_lag            : 1
    rebalance_frequency   : 1
    n_buckets             : 7
    initial_capital       : 70000000
    zscore_threshold      : 1.0
    lookback              : 20

======================================================================
Temporary dictory couldn't be removed manually.
Traceback (most recent call last):

  File pandas\_libs\tslibs\offsets.pyx:3878 in pandas._libs.tslibs.offsets._get_offset

KeyError: 'ME'


The above exception was the direct cause of the following exception:

Traceback (most recent call last):

  File pandas\_libs\tslibs\offsets.pyx:3979 in pandas._libs.tslibs.offsets.to_offset

  File pandas\_libs\tslibs\offsets.pyx:3884 in pandas._libs.tslibs.offsets._get_offset

ValueError: Invalid frequency: ME


The above exception was the direct cause of the following exception:

Traceback (most recent call last):

  File C:\ProgramData\anaconda3\lib\site-packages\spyder_kernels\py3compat.py:356 in compat_exec
    exec(code, globals, locals)

  File c:\users\mb.aliev\desktop\py_apps\!projects\prod_test.py:303
    fig_bt = plot_backtest_dashboard(result)

  File ~\Desktop\PY_apps\!projects\yield_curve\backtest.py:1161 in plot_backtest_dashboard
    monthly = result.equity_curve.resample("ME").last().pct_change().dropna() * 100

  File C:\ProgramData\anaconda3\lib\site-packages\pandas\core\series.py:5872 in resample
    return super().resample(

  File C:\ProgramData\anaconda3\lib\site-packages\pandas\core\generic.py:8858 in resample
    return get_resampler(

  File C:\ProgramData\anaconda3\lib\site-packages\pandas\core\resample.py:1543 in get_resampler
    tg = TimeGrouper(**kwds)

  File C:\ProgramData\anaconda3\lib\site-packages\pandas\core\resample.py:1613 in __init__
    freq = to_offset(freq)

  File pandas\_libs\tslibs\offsets.pyx:3891 in pandas._libs.tslibs.offsets.to_offset

  File pandas\_libs\tslibs\offsets.pyx:3987 in pandas._libs.tslibs.offsets.to_offset

ValueError: Invalid frequency: ME
