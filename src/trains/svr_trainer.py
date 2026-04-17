from sklearn.svm import SVR

from src.trains.base import BaseTrainer


class SVRTrainer(BaseTrainer):
    """Support Vector Regressor with RBF kernel.

    Kernel-based approach (different from gradient boosting).
    Can capture non-linear patterns that boosting methods might miss.
    Computationally more expensive but often provides complementary predictions for ensembles.
    """

    def __init__(
        self,
        C: float = 1000.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        epsilon: float = 0.1,
        timestamp_col: str = "timestamp",
        test_size: float = 0.2,
    ) -> None:
        super().__init__(timestamp_col, test_size)
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.epsilon = epsilon

    def build_model(self) -> SVR:
        return SVR(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            epsilon=self.epsilon,
        )
