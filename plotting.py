"""
yield_curve.plotting
====================

Модуль визуализации для библиотеки моделирования
рыночных кривых депозитных ставок.

Все функции возвращают plotly.graph_objects.Figure,
что обеспечивает совместимость с Jupyter, экспорт
в HTML/PNG и интеграцию с отчётами.

Функциональность
----------------
- Визуализация наблюдаемых и реконструированных кривых
- 3D-поверхность кривой во времени
- Компоненты PCA (Level, Slope, Curvature)
- Сходимость EM-алгоритма
- Scree-plot объяснённой дисперсии
- Heatmap остатков
- Визуализация PnL и directional accuracy
- Сводные дашборды

Автор:
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


###############################################################################
# Constants — единый визуальный стиль
###############################################################################

# Цветовая палитра (профессиональная, финансовая)
COLOR_OBSERVED = "#1f77b4"       # Синий — наблюдаемые данные
COLOR_FITTED = "#d62728"         # Красный — модель/реконструкция
COLOR_RESIDUAL_POS = "#2ca02c"   # Зелёный — положительные остатки
COLOR_RESIDUAL_NEG = "#d62728"   # Красный — отрицательные остатки
COLOR_COMPONENTS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
COLOR_PNL_POS = "#2ca02c"
COLOR_PNL_NEG = "#d62728"
COLOR_CONVERGENCE = "#1f77b4"

# Шаблон оформления
DEFAULT_TEMPLATE = "plotly_white"
DEFAULT_FONT_FAMILY = "Segoe UI, Arial, sans-serif"
DEFAULT_FONT_SIZE = 12
DEFAULT_TITLE_SIZE = 16

# Размеры по умолчанию
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 550

# Русские названия месяцев для форматирования дат
RU_MONTHS = [
    "Янв", "Фев", "Мар", "Апр", "Май", "Июн",
    "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек",
]


###############################################################################
# Style helpers
###############################################################################

def _apply_default_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Применение единого стиля ко всем графикам библиотеки.
    """

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=DEFAULT_TITLE_SIZE, family=DEFAULT_FONT_FAMILY),
            x=0.5,
            xanchor="center",
        ),
        template=DEFAULT_TEMPLATE,
        font=dict(family=DEFAULT_FONT_FAMILY, size=DEFAULT_FONT_SIZE),
        width=width,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified",
    )

    if xaxis_title:
        fig.update_xaxes(title_text=xaxis_title)
    if yaxis_title:
        fig.update_yaxes(title_text=yaxis_title)

    return fig


def save_figure(
    fig: go.Figure,
    path: Union[str, Path],
    format: str = "html",
    scale: int = 2,
) -> Path:
    """
    Сохранение фигуры в файл.

    Parameters
    ----------
    fig
        Plotly-фигура.
    path
        Путь к файлу (без расширения, если format задан).
    format
        'html' (интерактивный, для шаринга по сети),
        'png' (статичный, для вставки в Outlook),
        'json' (для повторной загрузки).
    scale
        Масштаб для растровых форматов (png).

    Returns
    -------
    Path
        Путь к сохранённому файлу.
    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "html":
        if path.suffix != ".html":
            path = path.with_suffix(".html")
        fig.write_html(str(path), include_plotlyjs="cdn")
    elif format == "png":
        if path.suffix != ".png":
            path = path.with_suffix(".png")
        fig.write_image(str(path), scale=scale)
    elif format == "json":
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        fig.write_json(str(path))
    else:
        raise ValueError(f"Unknown format: {format}. Use 'html', 'png' or 'json'.")

    return path


def _format_date_axis_ru(fig: go.Figure) -> go.Figure:
    """Форматирование оси X с русскими названиями месяцев."""
    fig.update_xaxes(
        tickformat="%d %b %Y",
    )
    return fig


###############################################################################
# Yield curve visualization
###############################################################################

def plot_curve_snapshot(
    curve: pd.DataFrame,
    date: Any,
    label: str = "Кривая",
    color: str = COLOR_OBSERVED,
    title: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Снимок кривой ставок на конкретную дату.

    Parameters
    ----------
    curve
        Матрица ставок в wide-формате (index=date, columns=buckets).
    date
        Дата среза.
    label
        Подпись линии в легенде.
    color
        Цвет линии.
    title
        Заголовок графика. Если None, формируется автоматически.

    Returns
    -------
    go.Figure
    """

    if date not in curve.index:
        raise ValueError(f"Date {date} not found in curve index.")

    row = curve.loc[date].dropna()

    if title is None:
        if hasattr(date, "strftime"):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)
        title = f"Кривая ставок на {date_str}"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=row.index.tolist(),
        y=row.values.tolist(),
        mode="lines+markers",
        name=label,
        line=dict(color=color, width=2),
        marker=dict(size=8),
    ))

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Срочность, дней",
        yaxis_title="Ставка, % годовых",
        width=width,
        height=height,
    )

    return fig


def plot_curves_comparison(
    observed: pd.DataFrame,
    fitted: pd.DataFrame,
    date: Any,
    title: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Сравнение наблюдаемой и реконструированной кривых на одну дату.

    Дополнительно показывает остатки (разность) на нижней панели.

    Parameters
    ----------
    observed
        Исходная матрица ставок.
    fitted
        Реконструированная матрица ставок.
    date
        Дата среза.
    title
        Заголовок.

    Returns
    -------
    go.Figure
    """

    if date not in observed.index or date not in fitted.index:
        raise ValueError(f"Date {date} not found in both curves.")

    obs_row = observed.loc[date].dropna()
    fit_row = fitted.loc[date].dropna()

    common_cols = obs_row.index.intersection(fit_row.index)
    if len(common_cols) == 0:
        raise ValueError("No common buckets between observed and fitted.")

    obs_v = obs_row.loc[common_cols]
    fit_v = fit_row.loc[common_cols]
    residual = obs_v - fit_v

    if title is None:
        if hasattr(date, "strftime"):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)
        title = f"Наблюдаемая vs Реконструированная: {date_str}"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Кривые ставок", "Остатки (наблюдаемая − модель)"),
    )

    # Верхняя панель — кривые
    fig.add_trace(go.Scatter(
        x=common_cols.tolist(),
        y=obs_v.values.tolist(),
        mode="lines+markers",
        name="Наблюдаемая",
        line=dict(color=COLOR_OBSERVED, width=2),
        marker=dict(size=8),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=common_cols.tolist(),
        y=fit_v.values.tolist(),
        mode="lines+markers",
        name="Модель",
        line=dict(color=COLOR_FITTED, width=2, dash="dash"),
        marker=dict(size=8),
    ), row=1, col=1)

    # Нижняя панель — остатки
    colors = [
        COLOR_RESIDUAL_POS if r >= 0 else COLOR_RESIDUAL_NEG
        for r in residual.values
    ]

    fig.add_trace(go.Bar(
        x=common_cols.tolist(),
        y=residual.values.tolist(),
        name="Остаток",
        marker_color=colors,
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Срочность, дней",
        yaxis_title="Ставка, % годовых",
        width=width,
        height=int(height * 1.3),
    )

    fig.update_yaxes(title_text="Остаток, п.п.", row=2, col=1)

    return fig


def plot_curve_surface(
    curve: pd.DataFrame,
    title: str = "Эволюция кривой ставок во времени",
    colorscale: str = "Viridis",
    width: int = DEFAULT_WIDTH,
    height: int = int(DEFAULT_HEIGHT * 1.3),
) -> go.Figure:
    """
    3D-поверхность кривой ставок во времени.

    Ось X — срочность (дни), ось Y — дата, ось Z — ставка.
    Полезна для визуальной оценки структурных сдвигов
    и выявления аномалий.

    Parameters
    ----------
    curve
        Матрица ставок (wide).
    title
    colorscale
        Цветовая шкала Plotly.
    width, height

    Returns
    -------
    go.Figure
    """

    dates = curve.index
    buckets = curve.columns.tolist()

    Z = curve.values.T  # shape: (buckets, dates)

    # Преобразуем даты в числовой формат для оси Y
    date_ordinal = np.array([d.toordinal() for d in dates])

    fig = go.Figure(data=go.Surface(
        x=buckets,
        y=date_ordinal,
        z=Z,
        colorscale=colorscale,
        colorbar=dict(title="Ставка, %", len=0.7),
        hovertemplate=(
            "Срочность: %{x} дн.<br>"
            "Дата: %{y}<br>"
            "Ставка: %{z:.3f}%<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=DEFAULT_TITLE_SIZE, family=DEFAULT_FONT_FAMILY),
            x=0.5,
        ),
        template=DEFAULT_TEMPLATE,
        font=dict(family=DEFAULT_FONT_FAMILY, size=DEFAULT_FONT_SIZE),
        width=width,
        height=height,
        scene=dict(
            xaxis_title="Срочность, дней",
            yaxis_title="Дата",
            zaxis_title="Ставка, % годовых",
            aspectmode="auto",
        ),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


def plot_curve_timeseries(
    curve: pd.DataFrame,
    buckets: Optional[Iterable[int]] = None,
    title: str = "Динамика ставок по срочностям",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Временные ряды ставок для выбранных бакетов.

    Показывает, как менялись ставки каждой срочности во времени.
    Удобно для оценки волатильности и структурных сдвигов.

    Parameters
    ----------
    curve
        Матрица ставок (wide).
    buckets
        Какие бакеты показать. Если None — все.
    title
    width, height

    Returns
    -------
    go.Figure
    """

    if buckets is None:
        buckets = curve.columns.tolist()
    else:
        buckets = list(buckets)

    fig = go.Figure()

    for i, bucket in enumerate(buckets):
        if bucket not in curve.columns:
            continue
        color = COLOR_COMPONENTS[i % len(COLOR_COMPONENTS)]
        fig.add_trace(go.Scatter(
            x=curve.index.tolist(),
            y=curve[bucket].values.tolist(),
            mode="lines",
            name=f"{bucket} дн.",
            line=dict(color=color, width=1.5),
        ))

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Дата",
        yaxis_title="Ставка, % годовых",
        width=width,
        height=height,
    )

    fig = _format_date_axis_ru(fig)

    return fig


###############################################################################
# PCA components
###############################################################################

def plot_components(
    components: pd.DataFrame,
    n_components: int = 3,
    title: str = "Главные компоненты кривой ставок",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Визуализация главных компонент (факторов) кривой.

    Классическая интерпретация Litterman-Scheinkman:
    - PC1: Level (параллельный сдвиг)
    - PC2: Slope (наклон)
    - PC3: Curvature (выпуклость)

    Parameters
    ----------
    components
        Матрица компонент: index=component_id, columns=buckets.
        Обычно это interpolator.get_components().
    n_components
        Сколько компонент показать.
    title
    width, height

    Returns
    -------
    go.Figure
    """

    n_show = min(n_components, len(components))
    labels = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
    labels += [f"PC{i+1}" for i in range(3, n_show)]

    fig = go.Figure()

    for i in range(n_show):
        comp = components.iloc[i]
        fig.add_trace(go.Scatter(
            x=comp.index.tolist(),
            y=comp.values.tolist(),
            mode="lines+markers",
            name=labels[i],
            line=dict(color=COLOR_COMPONENTS[i], width=2),
            marker=dict(size=6),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Срочность, дней",
        yaxis_title="Нагрузка",
        width=width,
        height=height,
    )

    return fig


def plot_scores_timeseries(
    scores: pd.DataFrame,
    n_components: int = 3,
    title: str = "Динамика факторных нагрузок",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Временные ряды факторных нагрузок (scores).

    Показывает, как во времени менялся вклад каждого фактора
    в формирование кривой.

    Parameters
    ----------
    scores
        Матрица scores: index=date, columns=component_id.
    n_components
    title
    width, height

    Returns
    -------
    go.Figure
    """

    n_show = min(n_components, scores.shape[1])
    labels = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
    labels += [f"PC{i+1}" for i in range(3, n_show)]

    fig = make_subplots(
        rows=n_show, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=labels[:n_show],
    )

    for i in range(n_show):
        col = scores.columns[i]
        fig.add_trace(go.Scatter(
            x=scores.index.tolist(),
            y=scores[col].values.tolist(),
            mode="lines",
            name=labels[i],
            line=dict(color=COLOR_COMPONENTS[i], width=1.5),
            showlegend=(i == 0),
        ), row=i+1, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=i+1, col=1)

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Дата",
        yaxis_title="Значение фактора",
        width=width,
        height=int(height * (0.6 + 0.4 * n_show)),
    )

    fig = _format_date_axis_ru(fig)

    return fig


###############################################################################
# Diagnostics plots
###############################################################################

def plot_convergence(
    engine: Any,
    title: str = "Сходимость EM-алгоритма",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    График сходимости EM-алгоритма по итерациям.

    Parameters
    ----------
    engine
        Обученный движок (например, EMPCAEngine).
    title
    width, height

    Returns
    -------
    go.Figure
    """

    history = getattr(engine, "error_history_", None)

    if history is None:
        raise ValueError("Движок не хранит историю ошибок.")

    history_arr = np.asarray(history).flatten()
    iters = np.arange(1, len(history_arr) + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iters.tolist(),
        y=history_arr.tolist(),
        mode="lines+markers",
        name="Ошибка реконструкции",
        line=dict(color=COLOR_CONVERGENCE, width=2),
        marker=dict(size=5),
    ))

    # Горизонтальная линия tolerance
    tol = getattr(engine, "tolerance", None)
    if tol is not None and tol > 0:
        fig.add_hline(
            y=tol,
            line_dash="dash",
            line_color="red",
            annotation_text=f"tolerance = {tol:.2e}",
            annotation_position="bottom right",
        )

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Итерация",
        yaxis_title="Ошибка",
        width=width,
        height=height,
    )

    fig.update_yaxes(type="log")

    return fig


def plot_scree(
    explained_variance: pd.Series,
    cumulative: bool = True,
    threshold: float = 0.95,
    title: str = "Объяснённая дисперсия по компонентам",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Scree-plot: доля и кумулятивная доля объяснённой дисперсии.

    Parameters
    ----------
    explained_variance
        Series с долей дисперсии по каждой компоненте.
    cumulative
        Показывать ли кумулятивную кривую.
    threshold
        Порог для горизонтальной линии.
    title
    width, height

    Returns
    -------
    go.Figure
    """

    fig = go.Figure()

    x = [f"PC{i+1}" for i in range(len(explained_variance))]

    # Столбцы — индивидуальные доли
    fig.add_trace(go.Bar(
        x=x,
        y=explained_variance.values.tolist(),
        name="Индивидуальная доля",
        marker_color=COLOR_OBSERVED,
    ))

    if cumulative:
        cumulative_v = explained_variance.cumsum()
        fig.add_trace(go.Scatter(
            x=x,
            y=cumulative_v.values.tolist(),
            mode="lines+markers",
            name="Кумулятивная доля",
            line=dict(color=COLOR_FITTED, width=2),
            marker=dict(size=8),
            yaxis="y2",
        ))

        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="gray",
            yref="y2",
            annotation_text=f"порог {threshold:.0%}",
            annotation_position="bottom right",
        )

        fig.update_layout(
            yaxis2=dict(
                title="Кумулятивная доля",
                overlaying="y",
                side="right",
                range=[0, 1.05],
                tickformat=".0%",
            ),
        )

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Компонента",
        yaxis_title="Доля дисперсии",
        width=width,
        height=height,
    )

    fig.update_yaxes(tickformat=".0%")

    return fig


def plot_residuals_heatmap(
    residuals: pd.DataFrame,
    title: str = "Heatmap остатков",
    colorscale: str = "RdBu_r",
    width: int = DEFAULT_WIDTH,
    height: int = int(DEFAULT_HEIGHT * 1.4),
) -> go.Figure:
    """
    Heatmap остатков (дата × бакет).

    Красный — модель занижает, синий — завышает.
    Позволяет быстро выявить систематические ошибки.

    Parameters
    ----------
    residuals
        Матрица остатков (wide).
    title
    colorscale
    width, height

    Returns
    -------
    go.Figure
    """

    fig = go.Figure(data=go.Heatmap(
        x=residuals.columns.tolist(),
        y=residuals.index.tolist(),
        z=residuals.values.T,
        colorscale=colorscale,
        zmid=0,
        colorbar=dict(title="Остаток, п.п.", len=0.7),
        hovertemplate=(
            "Дата: %{y}<br>"
            "Срочность: %{x} дн.<br>"
            "Остаток: %{z:.4f} п.п.<extra></extra>"
        ),
    ))

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Срочность, дней",
        yaxis_title="Дата",
        width=width,
        height=height,
    )

    return fig


def plot_residuals_by_bucket(
    residuals: pd.DataFrame,
    title: str = "Распределение остатков по бакетам",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Box-plot остатков по каждому бакету.

    Показывает медиану, разброс и выбросы.
    """

    fig = go.Figure()

    for bucket in residuals.columns:
        values = residuals[bucket].dropna().values
        fig.add_trace(go.Box(
            y=values.tolist(),
            name=str(bucket),
            boxmean="sd",
            marker_color=COLOR_OBSERVED,
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Срочность, дней",
        yaxis_title="Остаток, п.п.",
        width=width,
        height=height,
    )

    return fig


###############################################################################
# Metrics plots
###############################################################################

def plot_pnl_timeseries(
    pnl: pd.DataFrame,
    title: str = "Накопленный PnL от торговли на сигналах модели",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Накопленный PnL по времени.

    Parameters
    ----------
    pnl
        Результат metrics.pnl_per_bucket().
    title
    width, height

    Returns
    -------
    go.Figure
    """

    cumulative = pnl.sum(axis=1).cumsum()

    fig = go.Figure()

    # Заливка цветом в зависимости от знака
    colors = [
        COLOR_PNL_POS if v >= 0 else COLOR_PNL_NEG
        for v in cumulative.values
    ]

    fig.add_trace(go.Scatter(
        x=cumulative.index.tolist(),
        y=cumulative.values.tolist(),
        mode="lines",
        name="Накопленный PnL",
        line=dict(color=COLOR_OBSERVED, width=2),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.2)",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Дата",
        yaxis_title="PnL, ден. ед.",
        width=width,
        height=height,
    )

    fig = _format_date_axis_ru(fig)

    return fig


def plot_pnl_by_bucket(
    pnl: pd.DataFrame,
    title: str = "PnL по бакетам",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Суммарный PnL по каждому бакету.
    """

    total = pnl.sum(axis=0)

    colors = [
        COLOR_PNL_POS if v >= 0 else COLOR_PNL_NEG
        for v in total.values
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=total.index.tolist(),
        y=total.values.tolist(),
        marker_color=colors,
        name="PnL",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Срочность, дней",
        yaxis_title="Суммарный PnL, ден. ед.",
        width=width,
        height=height,
    )

    return fig


def plot_directional_accuracy_rolling(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    window: int = 30,
    title: str = "Скользящая направленная точность",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Скользящая directional accuracy.

    Показывает, как меняется способность модели предсказывать
    направление движения ставки во времени.

    Parameters
    ----------
    predicted, actual
        Кривые (wide).
    window
        Размер окна для скользящего среднего.
    title
    width, height

    Returns
    -------
    go.Figure
    """

    p_diff = predicted.diff(axis=0)
    a_diff = actual.diff(axis=0)

    # По каждой дате считаем долю верных предсказаний по бакетам
    records = []
    for date in predicted.index[1:]:
        if date not in a_diff.index:
            continue
        p_sign = np.sign(p_diff.loc[date].values)
        a_sign = np.sign(a_diff.loc[date].values)
        valid = ~(np.isnan(p_sign) | np.isnan(a_sign) | (a_sign == 0))
        if valid.sum() == 0:
            records.append({"date": date, "da": np.nan})
            continue
        correct = (p_sign[valid] == a_sign[valid]).mean()
        records.append({"date": date, "da": correct})

    da_series = pd.DataFrame(records).set_index("date")["da"]
    rolling_da = da_series.rolling(window, min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_da.index.tolist(),
        y=rolling_da.values.tolist(),
        mode="lines",
        name=f"DA (окно {window})",
        line=dict(color=COLOR_OBSERVED, width=2),
    ))

    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="случайное угадывание",
        annotation_position="bottom right",
    )

    fig = _apply_default_layout(
        fig,
        title=title,
        xaxis_title="Дата",
        yaxis_title="Доля верных предсказаний",
        width=width,
        height=height,
    )

    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig = _format_date_axis_ru(fig)

    return fig


###############################################################################
# 3D surface
###############################################################################
def plot_curve_surface_3d(
    curve: pd.DataFrame,
    title: str = "3D-поверхность кривой ставок",
    colorscale: str = "Viridis",
    show_contour: bool = True,
    width: int = 1200,
    height: int = 800,
) -> go.Figure:
    """
    Интерактивная 3D-поверхность кривой ставок.

    Оси:
    - X: срочность (дни)
    - Y: дата
    - Z: ставка (% годовых)

    Parameters
    ----------
    curve
        Матрица ставок в wide-формате (index=date, columns=buckets).
    title
    colorscale
        Цветовая шкала Plotly.
    show_contour
        Показывать ли контурные линии на основании.
    width, height
    """

    dates = curve.index
    buckets = curve.columns.tolist()

    # Z: shape (buckets, dates)
    Z = curve.values.T

    # Преобразуем даты в числовой формат для оси Y
    date_ordinal = np.array([d.toordinal() for d in dates])
    
    # Подписи дат для hover
    date_labels = [d.strftime("%Y-%m-%d") for d in dates]

    fig = go.Figure(data=go.Surface(
        x=buckets,
        y=date_ordinal,
        z=Z,
        colorscale=colorscale,
        colorbar=dict(
            title="Ставка, %",
            len=0.7,
            thickness=15,
        ),
        contours=go.surface.Contours(
            z=dict(
                show=show_contour,
                usecolormap=True,
                highlightcolor="lime",
                project_z=True,
            )
        ) if show_contour else None,
        hovertemplate=(
            "Срочность: %{x} дн.<br>"
            "Дата: %{customdata}<br>"
            "Ставка: %{z:.4f}%<extra></extra>"
        ),
        customdata=np.array([[d for d in date_labels] for _ in buckets]).T,
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=DEFAULT_TITLE_SIZE + 2, family=DEFAULT_FONT_FAMILY),
            x=0.5,
        ),
        template=DEFAULT_TEMPLATE,
        font=dict(family=DEFAULT_FONT_FAMILY, size=DEFAULT_FONT_SIZE),
        width=width,
        height=height,
        scene=dict(
            xaxis=dict(
                title="Срочность, дней",
                gridcolor="rgb(200, 200, 200)",
                zerolinecolor="rgb(200, 200, 200)",
            ),
            yaxis=dict(
                title="Дата",
                gridcolor="rgb(200, 200, 200)",
                zerolinecolor="rgb(200, 200, 200)",
                tickvals=date_ordinal[::max(1, len(date_ordinal)//10)],
                ticktext=date_labels[::max(1, len(date_labels)//10)],
            ),
            zaxis=dict(
                title="Ставка, % годовых",
                gridcolor="rgb(200, 200, 200)",
                zerolinecolor="rgb(200, 200, 200)",
            ),
            aspectmode="auto",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
            ),
        ),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


###############################################################################
# Dashboard
###############################################################################

def plot_full_dashboard(
    interpolator: Any,
    actual: Optional[pd.DataFrame] = None,
    date: Optional[Any] = None,
    width: int = 1400,
    height: int = 2400,
) -> go.Figure:
    """
    Сводный дашборд по обученному интерполятору.

    Включает:
    1. Сравнение наблюдаемой и модельной кривой (если есть actual)
    2. Главные компоненты
    3. Сходимость EM-алгоритма
    4. Scree-plot
    5. Heatmap остатков
    6. PnL (если есть actual)

    Parameters
    ----------
    interpolator
        Обученный YieldCurveInterpolator.
    actual
        Эталонная кривая (wide). Если передана, добавляются
        метрики сравнения.
    date
        Дата для среза. Если None — берётся последняя.
    width, height

    Returns
    -------
    go.Figure
    """

    if not getattr(interpolator, "is_fitted_", False):
        raise RuntimeError("Интерполятор не обучен.")

    from . import diagnostics as diag
    from . import metrics as met

    original = interpolator.original_curve_
    fitted = interpolator.curve_
    mask = interpolator.mask_

    if date is None:
        date = original.index[-1]

    # Определяем структуру дашборда
    has_actual = actual is not None
    n_rows = 5 if has_actual else 4

    fig = make_subplots(
        rows=n_rows, cols=2,
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
        subplot_titles=[
            "Наблюдаемая vs Модель",
            "Главные компоненты",
            "Сходимость EM-алгоритма",
            "Scree-plot",
            "Heatmap остатков",
            "Box-plot остатков",
        ] + (["Накопленный PnL", "PnL по бакетам"] if has_actual else []),
    )

    # 1. Сравнение кривых (row 1, col 1)
    if date in original.index and date in fitted.index:
        obs_row = original.loc[date].dropna()
        fit_row = fitted.loc[date].dropna()
        common = obs_row.index.intersection(fit_row.index)

        fig.add_trace(go.Scatter(
            x=common.tolist(), y=obs_row.loc[common].values.tolist(),
            mode="lines+markers", name="Наблюдаемая",
            line=dict(color=COLOR_OBSERVED, width=2),
            showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=common.tolist(), y=fit_row.loc[common].values.tolist(),
            mode="lines+markers", name="Модель",
            line=dict(color=COLOR_FITTED, width=2, dash="dash"),
            showlegend=True,
        ), row=1, col=1)

    # 2. Компоненты (row 1, col 2)
    components = interpolator.get_components()
    if components is not None:
        for i in range(min(3, len(components))):
            comp = components.iloc[i]
            fig.add_trace(go.Scatter(
                x=comp.index.tolist(), y=comp.values.tolist(),
                mode="lines+markers",
                name=f"PC{i+1}",
                line=dict(color=COLOR_COMPONENTS[i], width=2),
                showlegend=True,
            ), row=1, col=2)

    # 3. Сходимость (row 2, col 1)
    history = getattr(interpolator._engine, "error_history_", None)
    if history is not None:
        history_arr = np.asarray(history).flatten()
        iters = np.arange(1, len(history_arr) + 1)
        fig.add_trace(go.Scatter(
            x=iters.tolist(), y=history_arr.tolist(),
            mode="lines+markers", name="Ошибка",
            line=dict(color=COLOR_CONVERGENCE, width=2),
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(type="log", row=2, col=1)

    # 4. Scree-plot (row 2, col 2)
    ev = diag.explained_variance(original, fitted)
    cv = diag.cumulative_variance(original, fitted)
    if not ev.empty:
        x = [f"PC{i+1}" for i in range(len(ev))]
        fig.add_trace(go.Bar(
            x=x, y=ev.values.tolist(),
            name="Доля", marker_color=COLOR_OBSERVED,
            showlegend=False,
        ), row=2, col=2)
        if not cv.empty:
            fig.add_trace(go.Scatter(
                x=x, y=cv.values.tolist(),
                mode="lines+markers", name="Кумул.",
                line=dict(color=COLOR_FITTED, width=2),
                yaxis="y2", showlegend=False,
            ), row=2, col=2)

    # 5. Heatmap остатков (row 3, col 1)
    residuals = diag.compute_residuals(original, fitted, mask)
    fig.add_trace(go.Heatmap(
        x=residuals.columns.tolist(),
        y=residuals.index.tolist(),
        z=residuals.values.T,
        colorscale="RdBu_r", zmid=0,
        showscale=False, name="Остатки",
    ), row=3, col=1)

    # 6. Box-plot остатков (row 3, col 2)
    for bucket in residuals.columns:
        values = residuals[bucket].dropna().values
        fig.add_trace(go.Box(
            y=values.tolist(), name=str(bucket),
            marker_color=COLOR_OBSERVED, showlegend=False,
        ), row=3, col=2)

    row_offset = 0

    # 7-8. PnL (если есть actual)
    if has_actual:
        pnl = met.pnl_per_bucket(fitted, actual, mask)
        cumulative_pnl = pnl.sum(axis=1).cumsum()

        fig.add_trace(go.Scatter(
            x=cumulative_pnl.index.tolist(),
            y=cumulative_pnl.values.tolist(),
            mode="lines", name="PnL",
            line=dict(color=COLOR_OBSERVED, width=2),
            fill="tozeroy", showlegend=False,
        ), row=4, col=1)

        total_pnl = pnl.sum(axis=0)
        colors = [
            COLOR_PNL_POS if v >= 0 else COLOR_PNL_NEG
            for v in total_pnl.values
        ]
        fig.add_trace(go.Bar(
            x=total_pnl.index.tolist(),
            y=total_pnl.values.tolist(),
            marker_color=colors, name="PnL по бакетам",
            showlegend=False,
        ), row=4, col=2)

        row_offset = 1

    # Общий layout
    fig = _apply_default_layout(
        fig,
        title="Сводный дашборд модели кривой ставок",
        width=width,
        height=height,
    )

    fig.update_layout(
        title=dict(
            text="Сводный дашборд модели кривой ставок",
            font=dict(size=DEFAULT_TITLE_SIZE + 2, family=DEFAULT_FONT_FAMILY),
            x=0.5,
        ),
    )

    return fig


###############################################################################
# Convenience wrappers
###############################################################################

def show(
    fig: go.Figure,
    renderer: Optional[str] = None,
) -> None:
    """
    Отображение фигуры (обёртка над fig.show()).
    """
    fig.show(renderer=renderer)
