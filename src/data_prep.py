import enum
import glob
import json
from itertools import chain
from typing import Iterable, Literal, Sequence

import dateutil
import numpy as np
import pandas as pd
from scipy import interpolate

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


class VideoDataPreparer:
    def __init__(
        self,
        max_length: int = MAX_LENGTH,
        step_sec: int = STEP_SEC,
        target_sec: int = TARGET_SEC,
    ) -> None:
        self.max_length = max_length
        self.step_sec = step_sec
        self.target_sec = target_sec

    def json_to_df(
        self,
        path: str,
        value_cols: Sequence[str] = (CN.VIEWS,),
    ) -> pd.DataFrame:
        with open(path, "r") as f:
            j_data = json.load(f)

        df = pd.json_normalize(j_data["data"])
        df[CN.TIMESTAMP] = df[CN.TIMESTAMP].apply(
            lambda x: int(dateutil.parser.parse(x).timestamp())
        )
        df[CN.TIMESTAMP] = df[CN.TIMESTAMP].astype("int64")

        required_cols = [CN.TIMESTAMP] + list(value_cols)
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns are missing in source json: {missing_cols}")

        return df[required_cols]

    @staticmethod
    def normalize_df(
        df: pd.DataFrame,
        keys_y: list[str],
        key_x: str,
        diap: Iterable,
        method: Literal[
            "linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic"
        ] = "linear",
    ) -> pd.DataFrame:
        df_sorted = df.sort_values(key_x)
        x_new = np.array(list(diap))
        df_new = pd.DataFrame({key_x: x_new})

        for key_y in keys_y:
            f = interpolate.interp1d(
                df_sorted[key_x],
                df_sorted[key_y],
                kind=method,
                fill_value="extrapolate",
                bounds_error=False,
            )
            df_new[key_y] = f(x_new)

        return df_new

    def generate_video_row(
        self,
        path: str,
        include_target: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.json_to_df(path)

        begin = int(df[CN.TIMESTAMP].min())
        end = int(
            min(df[CN.TIMESTAMP].max(), begin + self.step_sec * (self.max_length + 0.5))
        )
        target_timestamp = begin + self.target_sec

        if include_target:
            df = df[df[CN.TIMESTAMP] < target_timestamp]

        if df[CN.VIEWS].isna().sum() != 0:
            raise ValueError(f"NaN values in views for file {path}")

        old_df = df.copy()

        if include_target:
            diap = chain(
                range(begin, end, self.step_sec),
                range(target_timestamp, target_timestamp + 1, 1),
            )
        else:
            diap = range(begin, end, self.step_sec)

        df = self.normalize_df(
            df,
            [CN.VIEWS],
            CN.TIMESTAMP,
            diap,
        )

        df = df[1:]
        df[CN.CREATION_TIME] = df[CN.TIMESTAMP].apply(lambda x: int(x - begin))

        return old_df, df

    def to_model_row(
        self, prepared_df: pd.DataFrame, include_target: bool = True
    ) -> dict[str, float]:
        df_sorted = prepared_df.sort_values(CN.CREATION_TIME)
        row_dict: dict[str, float] = {
            CN.TIMESTAMP: float(df_sorted[CN.TIMESTAMP].min())
        }

        for _, row in df_sorted.iterrows():
            creation_time = int(row[CN.CREATION_TIME])
            if not include_target and creation_time == self.target_sec:
                continue
            col_name = f"{CN.VIEWS}{creation_time}"
            row_dict[col_name] = float(row[CN.VIEWS])

        return row_dict

    def build_training_dataset(
        self,
        row_path_glob: str,
        out_csv_path: str | None = None,
    ) -> pd.DataFrame:
        files = sorted(glob.glob(row_path_glob))
        rows: list[dict[str, float]] = []

        for i, path in enumerate(files):
            print(f"[{i + 1}/{len(files)}] {path}")
            try:
                _, prepared_df = self.generate_video_row(path=path, include_target=True)
                rows.append(self.to_model_row(prepared_df, include_target=True))
            except Exception as exc:
                print(f"Skip file {path}: {exc}")

        if not rows:
            raise ValueError("No rows were prepared for training dataset")

        out_df = pd.DataFrame(rows)

        if out_csv_path:
            out_df.to_csv(out_csv_path, sep=";", index=False)

        return out_df

    def build_inference_row(self, path: str) -> pd.DataFrame:
        _, prepared_df = self.generate_video_row(path=path, include_target=False)
        row = self.to_model_row(prepared_df, include_target=False)
        return pd.DataFrame([row])
