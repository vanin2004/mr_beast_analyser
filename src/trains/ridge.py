from sklearn.linear_model import Ridge

from src.trains.base import BaseTrainer


class RidgeTrainer(BaseTrainer):
    def __init__(
        self,
        alpha: float = 1.0,
        timestamp_col: str = "timestamp",
        test_size: float = 0.2,
    ) -> None:
        super().__init__(timestamp_col, test_size)
        self.alpha = alpha

    def build_model(self) -> Ridge:
        return Ridge(alpha=self.alpha)
