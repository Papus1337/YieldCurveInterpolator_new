import pandas as pd
import numpy as np

# -------------------------------------------------------
# 1. Загрузка реальных данных
# -------------------------------------------------------
# Предполагаем, что у тебя есть CSV/Excel файлы
# Формат: date, term_bucket, rate, amount

df_train = pd.read_csv("data/auctions_2023.csv", parse_dates=["date"])
df_eval = pd.read_csv("data/auctions_2024.csv", parse_dates=["date"])
df_benchmark = pd.read_csv("data/cb_curve_2024.csv", parse_dates=["date"])

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
        "tolerance": 1e-7,
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

# Маска пропусков из интерполятора (для eval-периода)
mask_aligned = interpolator.mask_.loc[common_dates] if interpolator.mask_ is not None else None


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
