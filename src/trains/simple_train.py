from sklearn.linear_model import LinearRegression

from src.trains.base import BaseTrainer


class SimpleLinearTrainer(BaseTrainer):
    def build_model(self) -> LinearRegression:
        return LinearRegression()
