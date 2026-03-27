import numpy as np
import pandas as pd
from typing import List


def normalize_column(df: pd.DataFrame, x: str, y: List[str], x0: int, step: int, xmax: int) -> pd.DataFrame:
    """
    Возвращает новый DataFrame с равномерно расположенными значениями колонки `x`
    от `x0` до `xmax` (включительно) с шагом `step`. Для каждой колонки в `y`
    выполняется линейная интерполяция по соседним не нулевым и не-null значениям.

    Особенности:
    - Значения `0` в колонках y считаются отсутствующими и игнорируются при интерполяции.
    - Если для колонки y нет ни одного ненулевого значения — в результирующем столбце будут NaN.
    - Если только одно ненулевое значение — оно будет подставлено только в точку, где оно встречается.

    Параметры:
    - df: входной DataFrame
    - x: имя колонки, которая рассматривается как координата/параметр (должна быть числовой)
    - y: список имён колонок, которые нужно интерполировать
    - x0: начальное значение для сетки
    - step: шаг сетки (целое положительное)
    - xmax: максимальное значение сетки
    """
    if step <= 0:
        raise ValueError("step must be positive")

    df2 = df.copy()
    # Преобразуем x в числовой тип и удалим строки с некорректным x
    df2[x] = pd.to_numeric(df2[x], errors='coerce')
    df2 = df2[df2[x].notna()]

    if df2.shape[0] == 0:
        # Пустой вход — возвращаем пустую равномерную сетку
        new_x = np.arange(x0, xmax + 1, step, dtype=float)
        res = pd.DataFrame({x: new_x})
        for col in y:
            res[col] = np.nan
        return res

    # Агрегируем по x на случай дублирующихся x (берём среднее по y)
    grouped = df2.groupby(x, as_index=True)[y].mean()

    # Создаём новую равномерную сетку
    new_x = np.arange(x0, xmax + 1, step)

    # Результирующий DataFrame
    result = pd.DataFrame({x: new_x})

    # Для каждой y-колонки выполняем интерполяцию по числовому индексу
    for col in y:
        if col in grouped.columns:
            series = grouped[col].astype(float)
        else:
            # Если такой колонки нет во входном наборе — заполним NaN
            series = pd.Series(dtype='float64')

        # Считаем, что значения 0 и NaN — отсутствуют и не участвуют в интерполяции
        mask = series.notna() & (series != 0)

        if mask.sum() == 0:
            # Нет известных точек
            interp_vals = np.array([np.nan] * len(new_x), dtype=float)
        elif mask.sum() == 1:
            # Только одна точка — только в её x будет значение, в остальных NaN
            known_x = series[mask].index.to_numpy(dtype=float)
            known_y = series[mask].to_numpy(dtype=float)
            interp_vals = np.array([known_y[0] if xx == known_x[0] else np.nan for xx in new_x], dtype=float)
        else:
            known_x = series[mask].index.to_numpy(dtype=float)
            known_y = series[mask].to_numpy(dtype=float)
            # numpy.interp выполняет линейную интерполяцию, но для значений за пределами
            # известных точек возвращает крайние y. Поэтому явно ставим NaN за краями.
            interp_vals = np.interp(new_x.astype(float), known_x, known_y)
            left_mask = new_x < known_x.min()
            right_mask = new_x > known_x.max()
            interp_vals[left_mask | right_mask] = np.nan

        result[col] = interp_vals

    return result

