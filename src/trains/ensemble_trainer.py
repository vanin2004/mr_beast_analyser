import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.trains.base import BaseTrainer


class EnsembleTrainer(BaseTrainer):
    """Voting Ensemble combining LightGBM, XGBoost, and SVR.

    Combines three diverse models:
    - LightGBM (fast, accurate gradient boosting)
    - XGBoost (robust gradient boosting)
    - SVR (kernel-based, non-linear approach)

    Takes weighted average of predictions. Usually more stable and accurate
    than any single model as it captures different aspects of the data.
    """

    def __init__(
        self,
        lightgbm_weight: float = 0.5,
        xgboost_weight: float = 0.3,
        svr_weight: float = 0.2,
        timestamp_col: str = "timestamp",
        test_size: float = 0.2,
    ) -> None:
        super().__init__(timestamp_col, test_size)
        self.lightgbm_weight = lightgbm_weight
        self.xgboost_weight = xgboost_weight
        self.svr_weight = svr_weight

    def build_model(self) -> VotingRegressor:
        """Build ensemble with three diverse regressors."""

        lgbm_model = lgb.LGBMRegressor(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            max_depth=-1,
            min_data_in_leaf=20,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

        xgb_model = XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )

        svr_model = SVR(
            C=1000.0,
            kernel="rbf",
            gamma="scale",
            epsilon=0.1,
        )

        return VotingRegressor(
            estimators=[
                ("lightgbm", lgbm_model),
                ("xgboost", xgb_model),
                ("svr", svr_model),
            ],
            weights=[self.lightgbm_weight, self.xgboost_weight, self.svr_weight],
        )
