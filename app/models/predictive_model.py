"""
Module for training, evaluating, and persisting regression models using Polars DataFrames and various ML libraries.
"""
import os

# Third-party imports
import polars as pl
import joblib
from typing import Any, Dict, Tuple

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

MODEL_DIR = "app/models/artifacts"
AUX_DIR = "app/models/auxiliary"

class PredictiveModel:
    def train_temporal(self, X_train: pl.DataFrame, y_train: pl.Series,
                      X_test: pl.DataFrame = None, y_test: pl.Series = None) -> None:
        """
        Trains the model with temporally ordered data (no shuffling).
        Args:
            X_train (pl.DataFrame): Training features.
            y_train (pl.Series): Training target.
            X_test (pl.DataFrame, optional): Test features.
            y_test (pl.Series, optional): Test target.
        """
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.model.fit(self.X_train, self.y_train)
        if X_test is not None and y_test is not None:
            self.X_test = X_test.to_numpy()
            self.y_test = y_test.to_numpy()
        else:
            self.X_test = None
            self.y_test = None
            
    @staticmethod
    def split_temporal(df: pl.DataFrame, target_col: str, date_col: str = "Date"):
        """
        Splits the DataFrame ordered by date into three equal parts: train, test, forecast.
        Args:
            df (pl.DataFrame): Full DataFrame.
            target_col (str): Target column name.
            date_col (str): Date column name.
        Returns:
            (X_train, y_train, X_test, y_test, X_forecast): Tuple of sets.
        """
        df_sorted = df.sort(date_col)
        n = df_sorted.height
        t1 = n // 3
        t2 = 2 * n // 3
        X_train = df_sorted.slice(0, t1).drop(target_col)
        y_train = df_sorted.slice(0, t1)[target_col]
        X_test = df_sorted.slice(t1, t2 - t1).drop(target_col)
        y_test = df_sorted.slice(t1, t2 - t1)[target_col]
        X_forecast = df_sorted.slice(t2, n - t2).drop(target_col)
        return X_train, y_train, X_test, y_test, X_forecast
    
    def train(self, X: pl.DataFrame, y: pl.Series, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Randomly splits data, trains the model, stores train/test sets for evaluation.
        Args:
            X (pl.DataFrame): Feature matrix.
            y (pl.Series): Target variable.
            test_size (float): Fraction for test split.
            random_state (int): Seed for reproducibility.
        """
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_np, y_np, test_size=test_size, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> Tuple[Dict[str, float], Any]:
        """
        Evaluates the model on the stored test set and returns metrics and predictions.
        Returns:
            metrics (dict): Evaluation metrics (MSE, MAE, R2).
            predictions (np.ndarray): Predicted values for test set.
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test set available.")
        y_pred = self.model.predict(self.X_test)
        metrics = {
            "MSE": mean_squared_error(self.y_test, y_pred),
            "MAE": mean_absolute_error(self.y_test, y_pred),
            "R2": r2_score(self.y_test, y_pred)
        }
        return metrics, y_pred

    def predict(self, X: pl.DataFrame) -> Any:
        """
        Predicts target values for new data (generic prediction).
        Args:
            X (pl.DataFrame): Feature matrix.
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X.to_numpy())

    def forecast(self, X_future: pl.DataFrame) -> Any:
        """
        Predicts target values for future data (no target available).
        Args:
            X_future (pl.DataFrame): Future features.
        Returns:
            np.ndarray: Forecasted values.
        """
        X_np = X_future.to_numpy()
        if X_np.shape[0] == 0:
            import numpy as np
            return np.array([])
        return self.model.predict(X_np)
    
    """
    Class for training, evaluating, and persisting regression models using Polars DataFrames.
    """
    def __init__(self, model_type: str, params: Dict[str, Any]):
        """
        Initializes the predictive model.
        Args:
            model_type (str): Type of regression model.
            params (dict): Model hyperparameters.
        """
        self.model_type = model_type
        self.params = params
        self.model = self._initialize_model()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _initialize_model(self) -> Any:
        """
        Initializes the regression model based on type.
        Returns:
            Model instance.
        """
        if self.model_type == "Decision Tree":
            return DecisionTreeRegressor(**self.params)
        elif self.model_type == "XGBoost":
            return XGBRegressor(objective="reg:squarederror", **self.params)
        elif self.model_type == "Random Forest":
            return RandomForestRegressor(**self.params)
        elif self.model_type == "LightGBM":
            return LGBMRegressor(**self.params)
        elif self.model_type == "CatBoost":
            return CatBoostRegressor(
                **self.params,
                logging_level="Silent",
                train_dir=AUX_DIR
                )
        else:
            raise ValueError("Unsupported model type")

    def train(self, X: pl.DataFrame, y: pl.Series) -> None:
        """
        Splits data, trains the model.
        Args:
            X (pl.DataFrame): Feature matrix.
            y (pl.Series): Target variable.
        """
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> Tuple[Dict[str, float], Any]:
        """
        Evaluates the model and returns metrics and predictions.
        Returns:
            metrics (dict): Evaluation metrics.
            predictions (np.ndarray): Predicted values for test set.
        """
        y_pred = self.model.predict(self.X_test)
        metrics = {
            "MSE": mean_squared_error(self.y_test, y_pred),
            "MAE": mean_absolute_error(self.y_test, y_pred),
            "R2": r2_score(self.y_test, y_pred)
        }
        return metrics, y_pred

    def predict(self, X: pl.DataFrame) -> Any:
        """
        Predicts target values for new data.
        Args:
            X (pl.DataFrame): Feature matrix.
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X.to_numpy())

    def get_feature_importance(self, feature_names) -> Dict[str, float]:
        """
        Returns feature importances if available.
        Args:
            feature_names (list): List of feature names.
        Returns:
            dict: Feature importances.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))
        return None

    def save(self, filename: str) -> None:
        """
        Saves the trained model and its type to disk in the default model directory.
        Args:
            filename (str): Filename for the model artifact.
        """
        import os
        os.makedirs(MODEL_DIR, exist_ok=True)
        # If filename is a path, use as is; else, prepend MODEL_DIR
        if not (filename.startswith(MODEL_DIR) or filename.startswith("/")):
            path = os.path.join(MODEL_DIR, filename)
        else:
            path = filename
        import joblib
        joblib.dump({'model': self.model, 'model_type': self.model_type, 'params': self.params}, path)

    @classmethod
    def load(cls, filename: str) -> 'PredictiveModel':
        """
        Loads a trained model and its type from disk in the default model directory.
        Args:
            filename (str): Filename for the model artifact.
        Returns:
            PredictiveModel instance.
        """
        import os
        # If filename is a path, use as is; else, prepend MODEL_DIR
        if not (filename.startswith(MODEL_DIR) or filename.startswith("/")):
            path = os.path.join(MODEL_DIR, filename)
        else:
            path = filename
        import joblib
        data = joblib.load(path)
        instance = cls(model_type=data.get('model_type', 'Decision Tree'), params=data.get('params', {}))
        instance.model = data['model']
        return instance

    @staticmethod
    def save_aux_file(obj, filename: str) -> None:
        """
        Saves auxiliary files (e.g., encoders, scalers) in the auxiliary directory.
        Args:
            obj: Object to save.
            filename (str): Filename for the auxiliary file.
        """
        import os
        filename = os.path.basename(filename)
        os.makedirs(AUX_DIR, exist_ok=True)
        path = os.path.join(AUX_DIR, filename)
        import joblib
        joblib.dump(obj, path)

    @staticmethod
    def load_aux_file(filename: str):
        """
        Loads auxiliary files from the auxiliary directory.
        Args:
            filename (str): Filename for the auxiliary file.
        Returns:
            Loaded object.
        """
        import os
        filename = os.path.basename(filename)
        path = os.path.join(AUX_DIR, filename)
        import joblib
        return joblib.load(path)
