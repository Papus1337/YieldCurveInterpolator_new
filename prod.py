# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:01:56 2026

@author: mb.aliev


Моделирование кривой ставок и сохранение результата в БД.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

from yield_curve import preprocessing as pp
from yield_curve import YieldCurveInterpolator
from yield_curve import plotting


def DB_request(sql):
    server = 'trading-db.ahml1.ru'
    database = 'LIQUIDITY'
    
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';'
        'DATABASE=' + database + ';Trusted_connection=yes'
    )
    cursor = conn.cursor()
    
    t1 = time.time()
    try:
        df = pd.read_sql(sql, conn)
        print('Request completed: ', time.time() - t1)
    except Exception as e:
        print('Request execution error:', e)
    finally:
        cursor.close()
        conn.close()
    
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    return df


# -------------------------------------------------------
# 1. Загрузка исторических данных
# -------------------------------------------------------
sql_history = """
    select [date], TERM_AVG term_bucket, spread rate, amount
    from [LIQUIDITY].[qt].prev_spreads_from_auc_V2_2_WithPERC 
    where 1 = 1
    order by [date], term_avg
"""
df_history = DB_request(sql_history)

# -------------------------------------------------------
# 2. Параметры
# -------------------------------------------------------
BUCKETS = [1, 7, 14, 31, 61, 91, 181]
TARGET_DATE = pd.Timestamp('2026-06-26')  # Дата, на которую моделируем
FORECAST_DAYS = 7                          # На сколько дней прогнозируем

# Создаём директорию для отчётов
Path("reports").mkdir(exist_ok=True)

# -------------------------------------------------------
# 3. Подготовка данных
# -------------------------------------------------------
# Обучающая выборка — всё до целевой даты
df_train = df_history[df_history['date'] < TARGET_DATE].copy()

# Данные за целевую дату (для трансформации)
df_target = df_history[df_history['date'] == TARGET_DATE].copy()

print(f"\n=== Данные ===")
print(f"Train: {df_train.shape[0]} строк, период {df_train['date'].min()} — {df_train['date'].max()}")
print(f"Target date: {TARGET_DATE.date()}, {df_target.shape[0]} строк")
print(f"Пропусков в target: {df_target['rate'].isna().sum()}")

# Валидация
pp.validate_input(df_train, buckets=BUCKETS)
pp.validate_input(df_target, buckets=BUCKETS)

# -------------------------------------------------------
# 4. Обучение модели
# -------------------------------------------------------
interpolator = YieldCurveInterpolator(
    engine="empca",
    engine_params={
        "n_components": 6,
        "max_iter": 1000,
        "tol": 1e-6,
        "relaxation": 0.8,
    },
    buckets=BUCKETS,
    init_method="interpolate",
    scale=True,
)

print("\n=== Обучение модели ===")
interpolator.fit(df_train)

from yield_curve import diagnostics
summary = interpolator._engine.summary()
conv = diagnostics.check_convergence(summary)
print(f"Сходимость: {conv['verdict']}")
print(f"Итераций: {conv['n_iter']}")
print(f"Финальная ошибка: {conv['final_error']:.2e}")

# -------------------------------------------------------
# 5. Моделирование кривой на целевую дату
# -------------------------------------------------------
# Трансформируем данные за целевую дату через обученную модель
# На выходе — полная кривая без пропусков
target_curve = interpolator.transform(df_target)

print(f"\n=== Результат моделирования ===")
print(f"Кривая на {TARGET_DATE.date()}:")
print(target_curve.to_string())

print('\n=== Сравнение: модель vs простое среднее ===')
print('\n=== Разница: ===')
print((target_curve.iloc[0] - interpolator._columns_means_).to_string())

# -------------------------------------------------------
# 6. Прогноз на несколько дней вперёд
# -------------------------------------------------------
# Простой подход: используем последнюю кривую как базу
# и добавляем небольшой дрейф на основе тренда

def simple_forecast(
    interpolator,
    df_target,
    n_days=7,
):
    """
    Простой прогноз: берём кривую за последний день и добавляем
    дрейф, рассчитанный по последним N дням истории.
    """
    # Получаем реконструированную кривую за последний день
    target_curve = interpolator.transform(df_target)
    last_curve = target_curve.iloc[0].values
    
    # Рассчитываем дрейф по истории
    # Берём последние 20 дней истории
    train_curve = interpolator.get_curve()
    lookback = min(20, len(train_curve))
    recent = train_curve.iloc[-lookback:]
    
    # Среднее изменение за день
    daily_changes = recent.diff().iloc[1:]
    drift = daily_changes.mean().values
    
    # Генерируем будущие даты (только рабочие дни)
    future_dates = pd.bdate_range(
        start=df_target['date'].iloc[0] + pd.Timedelta(days=1),
        periods=n_days,
    )
    
    # Прогноз
    forecast_values = np.zeros((n_days, len(BUCKETS)))
    for i in range(n_days):
        forecast_values[i] = last_curve + drift * (i + 1)
    
    forecast_df = pd.DataFrame(
        forecast_values,
        index=future_dates,
        columns=BUCKETS,
    )
    
    return forecast_df


forecast_curve = simple_forecast(interpolator, df_target, n_days=FORECAST_DAYS)

print(f"\n=== Прогноз на {FORECAST_DAYS} дней ===")
print(forecast_curve.to_string())

# -------------------------------------------------------
# 7. Объединяем результат: целевая дата + прогноз
# -------------------------------------------------------
# Добавляем целевую дату в начало прогноза
full_result = pd.concat([target_curve, forecast_curve])
full_result = full_result.sort_index()

print(f"\n=== Итоговый результат ===")
print(f"Всего дат: {len(full_result)}")
print(full_result.to_string())

# -------------------------------------------------------
# 8. Визуализация
# -------------------------------------------------------

# 3D-поверхность: история + прогноз
# Берём последние 60 дней истории + прогноз
history_tail = interpolator.get_curve().iloc[-60:]
combined_for_plot = pd.concat([history_tail, forecast_curve])

fig_3d = plotting.plot_curve_surface_3d(
    combined_for_plot,
    title=f"Кривая ставок: история + прогноз на {FORECAST_DAYS} дней",
)
fig_3d.show()
plotting.save_figure(fig_3d, "reports/curve_3d_surface.html", format="html")

# Снимок кривой на целевую дату
fig_snapshot = plotting.plot_curve_snapshot(
    full_result,
    date=TARGET_DATE,
    label="Модель (восстановленная)",
    color="#1f77b4",
    title=f"Кривая ставок на {TARGET_DATE.date()}",
)
fig_snapshot.show()
plotting.save_figure(fig_snapshot, "reports/curve_snapshot.html", format="html")

# Динамика ставок по бакетам (история + прогноз)
fig_ts = plotting.plot_curve_timeseries(
    combined_for_plot,
    title="Динамика ставок: история + прогноз",
)
fig_ts.show()
plotting.save_figure(fig_ts, "reports/curve_timeseries.html", format="html")

print("\nОтчёты сохранены в директорию reports/")

# -------------------------------------------------------
# 9. Сохранение в БД
# -------------------------------------------------------
def save_to_db(df_wide, table_name, server, database, schema="qt"):
    """
    Сохранение wide-DataFrame в БД в long-формате.
    """
    # Конвертируем в long-формат
    records = []
    for date in df_wide.index:
        for bucket in df_wide.columns:
            rate = df_wide.loc[date, bucket]
            if pd.notna(rate):
                records.append({
                    "date": date,
                    "term_bucket": int(bucket),
                    "rate": float(rate),
                })
    
    if not records:
        print("Нет данных для сохранения.")
        return
    
    df_to_save = pd.DataFrame(records)
    
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_connection=yes"
    )
    
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    try:
        # Удаляем старые прогнозы на эти даты (если есть)
        dates_str = ",".join([f"'{d.strftime('%Y-%m-%d')}'" for d in df_wide.index])
        cursor.execute(f"""
            DELETE FROM [{schema}].[{table_name}]
            WHERE [date] IN ({dates_str})
        """)
        
        # Вставляем новые данные
        for _, row in df_to_save.iterrows():
            cursor.execute(
                f"""
                INSERT INTO [{schema}].[{table_name}] 
                ([date], [term_bucket], [rate])
                VALUES (?, ?, ?)
                """,
                row["date"],
                row["term_bucket"],
                row["rate"],
            )
        
        conn.commit()
        print(f"\nСохранено {len(records)} записей в [{schema}].[{table_name}]")
        
    except Exception as e:
        conn.rollback()
        print(f"Ошибка сохранения в БД: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

'''
# Сохраняем результат (целевая дата + прогноз)
save_to_db(
    df_wide=full_result,
    table_name="forecast_yield_curve",  # Имя таблицы для прогнозов
    server="trading-db.ahml1.ru",
    database="LIQUIDITY",
    schema="qt",
)

print("\n=== Готово ===")
#'''




runfile('C:/Users/mb.aliev/Desktop/PY_apps/!projects/prod.py', wdir='C:/Users/mb.aliev/Desktop/PY_apps/!projects')
Reloaded modules: yield_curve.preprocessing, yield_curve.engines.empca, yield_curve.engines, yield_curve.diagnostics, yield_curve.metrics, yield_curve, yield_curve.plotting, yield_curve.interpolator
Request completed:  0.03699541091918945

=== Данные ===
Train: 7287 строк, период 2022-04-08 00:00:00 — 2026-06-25 00:00:00
Target date: 2026-06-26, 7 строк
Пропусков в target: 3

=== Обучение модели ===
Сходимость: FAIL
Итераций: 1000
Финальная ошибка: 1.17e-04
Traceback (most recent call last):

  File C:\ProgramData\anaconda3\lib\site-packages\spyder_kernels\py3compat.py:356 in compat_exec
    exec(code, globals, locals)

  File c:\users\mb.aliev\desktop\py_apps\!projects\prod.py:122
    target_curve = interpolator.transform(df_target)

  File ~\Desktop\PY_apps\!projects\yield_curve\interpolator.py:192 in transform
    scaled, mask, original_curve = self._preprocess(df, is_fit=False)

  File ~\Desktop\PY_apps\!projects\yield_curve\interpolator.py:107 in _preprocess
    for col in initialized.columns():

TypeError: 'Int64Index' object is not callable

                
