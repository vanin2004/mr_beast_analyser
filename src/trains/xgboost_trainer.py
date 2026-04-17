from xgboost import XGBRegressor

from src.trains.base import BaseTrainer


class XGBoostTrainer(BaseTrainer):
    """XGBoost Regressor trainer - robust gradient boosting framework.

    More conservative than LightGBM, stable for production use.
    Good balance between accuracy and computational efficiency.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        random_state: int = 42,
        timestamp_col: str = "timestamp",
        test_size: float = 0.2,
        verbosity: int = 0,
    ) -> None:
        super().__init__(timestamp_col, test_size)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.verbosity = verbosity

    def build_model(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            verbosity=self.verbosity,
            n_jobs=-1,
        )
