from sklearn.ensemble import RandomForestRegressor

from src.trains.base import BaseTrainer


class RandomForestTrainer(BaseTrainer):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float = "sqrt",
        random_state: int = 42,
        timestamp_col: str = "timestamp",
        test_size: float = 0.2,
    ) -> None:
        super().__init__(timestamp_col, test_size)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
