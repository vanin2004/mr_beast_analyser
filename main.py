import datetime
import enum
import glob
from dataclasses import dataclass
import json
import dateutil

from typing import TypedDict
from collections import namedtuple
import numpy as np
import pandas as pd
from typing import List

from scipy import interpolate
from typing import Iterable

from itertools import chain



"""
{
  "videoId": "str",
  "data": [
    {
      "timestamp": "datetime(iso 8601 UTC)",
      "views": "int",
      "likes": "int",
      "comments": "int",
      "vph": "float"
    }
    ]
 }
"""


MAX_LENGTH = 24
STEP_SEC = 60 * 60
TARGET_SEC = 60 * 60 * 24 * 7

class CN(str, enum.Enum):
    VIDEO_ID = "videoId"
    TIMESTAMP = "timestamp"
    VIEWS = "views"
    LIKES = "likes"
    COMMENTS = "comments"
    CREATION_TIME = "creationTime"


def json_to_df(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        j_data = json.load(f)

    df = pd.json_normalize(j_data["data"])

    df[CN.TIMESTAMP] = df[CN.TIMESTAMP].apply(
        lambda x: int(dateutil.parser.parse(x).timestamp()))

    df[CN.TIMESTAMP].astype('int64')

    return df[[CN.TIMESTAMP, CN.VIEWS]]


def normalize_df(
        df: pd.DataFrame,
        keys_y: list[str],
        key_x: str,
        diap: Iterable,
        method: str = 'linear'
) -> pd.DataFrame:
    df_sorted = df.sort_values(key_x)

    x_new = np.array(list(diap))

    df_new = pd.DataFrame({key_x: x_new})

    for key_y in keys_y:
        f = interpolate.interp1d(
            df_sorted[key_x], df_sorted[key_y],
            kind=method, fill_value='extrapolate', bounds_error=False
        )
        df_new[key_y] = f(x_new)

    return df_new


def generate_video_row(path, MAX_LENGTH, STEP_SEC, TARGET_SEC):
    df = json_to_df(path)

    begin = int(df[CN.TIMESTAMP].min())
    end = int(min(df[CN.TIMESTAMP].max(), begin + STEP_SEC * (MAX_LENGTH + 0.5)))
    target_timestamp = begin + TARGET_SEC
    df = df[df[CN.TIMESTAMP] < target_timestamp]

    if df[CN.VIEWS].isna().sum() !=0:
        print(path)
        raise Exception

    old_df = df.copy()

    diap = chain(
        range(begin, end, STEP_SEC),
        range(target_timestamp, target_timestamp + 1, 1),
    )

    df = normalize_df(
        df,
        [CN.VIEWS],
        CN.TIMESTAMP,
        diap,
    )

    df = df[1:]

    df[CN.CREATION_TIME] = df[CN.TIMESTAMP].apply(lambda x: int(x-begin))

    return old_df, df

def load_data(row_path: str) -> None:
    files = glob.glob(row_path)
    first = True
    f = open("./data/out.csv", "w")
    for i, path in enumerate(files):
        print(i)
        try:
            old_df, df = generate_video_row(path, MAX_LENGTH, STEP_SEC, TARGET_SEC)
        except:
            continue


        df_sorted = df.sort_values(CN.CREATION_TIME)

        # print(df_sorted.dtypes)

        row_dict = {
            CN.TIMESTAMP: df_sorted[CN.TIMESTAMP].min()  # или первый timestamp
        }

        for _, row in df_sorted.iterrows():
            col_name = f"{CN.VIEWS}{int(row[CN.CREATION_TIME])}"
            row_dict[col_name] = row[CN.VIEWS]


        if first:
            print(df_sorted.dtypes)
            first = False
            f.write(";".join(row_dict.keys())+"\n")
        f.write(";".join(map(str,row_dict.values())) + "\n")

    f.close()

out_csv_path = "./data/out.csv"

# load_data("./data/row/*", out_csv_path)

df = pd.read_csv(out_csv_path, delimiter=";")

print(df.head())
print(df.dtypes)
