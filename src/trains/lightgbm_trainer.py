import lightgbm as lgb

from src.trains.base import BaseTrainer


class LightGBMTrainer(BaseTrainer):
    """LightGBM Regressor trainer - gradient boosting with efficient memory usage.

    Typically faster and more accurate than RandomForest on tabular data with many features.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        random_state: int = 42,
        timestamp_col: str = "timestamp",
        test_size: float = 0.2,
        verbose: int = -1,
    ) -> None:
        super().__init__(timestamp_col, test_size)
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.random_state = random_state
        self.verbose = verbose

    def build_model(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_data_in_leaf=self.min_data_in_leaf,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=-1,
        )
