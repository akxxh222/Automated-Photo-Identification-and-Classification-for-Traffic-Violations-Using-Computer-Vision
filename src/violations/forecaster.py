import json
import logging
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


LOGGER = logging.getLogger(__name__)


class TrafficForecaster:
    """Train and serve a lightweight violation-volume forecaster."""

    def __init__(
        self,
        dataset_path="data/datasets/Datasets/Indian Traffic Violation Kaggle.zip",
        model_path="models/traffic_forecaster.joblib",
        loc_path="configs/camera_locations.json",
    ):
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.metrics_path = self.model_path.with_suffix(".metrics.json")

        with open(loc_path, "r", encoding="utf-8") as f:
            self.locations = json.load(f)

        self.junction_zones = {}
        for cam, data in self.locations.items():
            junction_id = data.get("junction_id")
            if junction_id:
                self.junction_zones[junction_id] = data.get("zone_type", "standard")

        self.zone_multipliers = {
            "standard": 1.0,
            "school": 1.25,
            "hospital": 1.35,
        }
        self.feature_columns = [
            "hour",
            "dayofweek",
            "month",
            "dayofyear",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "doy_sin",
            "doy_cos",
            "is_weekend",
            "is_rush_hour",
        ]

        self.model = None
        self.global_average = 0.0
        self.seasonal_profile = {}
        self.residual_std = 1.0
        self.peak_threshold = 1.0
        self.validation_metrics = {
            "MAPE": None,
            "RMSE": None,
            "samples": 0,
            "status": "untrained",
        }

        self._load_artifact()

    def _load_artifact(self):
        if not self.model_path.exists():
            return

        try:
            artifact = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover - defensive loading path
            LOGGER.warning("Could not load traffic forecaster artifact: %s", exc)
            return

        self.model = artifact.get("model")
        self.global_average = artifact.get("global_average", 0.0)
        self.seasonal_profile = artifact.get("seasonal_profile", {})
        self.residual_std = artifact.get("residual_std", 1.0)
        self.peak_threshold = max(1.0, artifact.get("peak_threshold", 0.0))
        self.validation_metrics = artifact.get("validation_metrics", self.validation_metrics)

    def _read_dataset(self):
        if self.dataset_path.exists() and self.dataset_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(self.dataset_path, "r") as zf:
                csv_name = next(
                    (name for name in zf.namelist() if name.lower().endswith(".csv")),
                    None,
                )
                if csv_name is None:
                    raise FileNotFoundError(f"No CSV found inside {self.dataset_path}")
                with zf.open(csv_name) as f:
                    return pd.read_csv(f)

        if self.dataset_path.exists() and self.dataset_path.suffix.lower() == ".csv":
            return pd.read_csv(self.dataset_path)

        fallback = Path("data/datasets/Datasets/Indian Traffic Violation Kaggle.zip")
        if fallback.exists():
            with zipfile.ZipFile(fallback, "r") as zf:
                csv_name = next(
                    (name for name in zf.namelist() if name.lower().endswith(".csv")),
                    None,
                )
                if csv_name is None:
                    raise FileNotFoundError(f"No CSV found inside {fallback}")
                with zf.open(csv_name) as f:
                    return pd.read_csv(f)

        raise FileNotFoundError(
            "Could not find the Indian Traffic Violation dataset under data/datasets/Datasets."
        )

    @staticmethod
    def _build_features(timestamps: pd.Series) -> pd.DataFrame:
        ts = pd.to_datetime(timestamps)
        frame = pd.DataFrame({"timestamp": ts})
        frame["hour"] = frame["timestamp"].dt.hour
        frame["dayofweek"] = frame["timestamp"].dt.dayofweek
        frame["month"] = frame["timestamp"].dt.month
        frame["dayofyear"] = frame["timestamp"].dt.dayofyear
        frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype(int)
        frame["is_rush_hour"] = frame["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

        frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24.0)
        frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24.0)
        frame["dow_sin"] = np.sin(2 * np.pi * frame["dayofweek"] / 7.0)
        frame["dow_cos"] = np.cos(2 * np.pi * frame["dayofweek"] / 7.0)
        frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12.0)
        frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12.0)
        frame["doy_sin"] = np.sin(2 * np.pi * frame["dayofyear"] / 366.0)
        frame["doy_cos"] = np.cos(2 * np.pi * frame["dayofyear"] / 366.0)

        return frame

    def _build_training_table(self):
        raw = self._read_dataset()
        if "Date" not in raw.columns or "Time" not in raw.columns:
            raise ValueError("The violation CSV must contain Date and Time columns.")

        dt = pd.to_datetime(
            raw["Date"].astype(str).str.strip() + " " + raw["Time"].astype(str).str.strip(),
            errors="coerce",
        )
        hourly = pd.DataFrame({"timestamp": dt}).dropna()
        if hourly.empty:
            raise ValueError("No parsable timestamps were found in the violation CSV.")

        hourly["timestamp"] = hourly["timestamp"].dt.floor("h")
        counts = hourly.groupby("timestamp").size().rename("count")
        full_index = pd.date_range(counts.index.min(), counts.index.max(), freq="h")
        counts = counts.reindex(full_index, fill_value=0)
        table = counts.reset_index().rename(columns={"index": "timestamp"})
        table = self._build_features(table["timestamp"])
        table["count"] = counts.values.astype(float)
        return table

    def _profile_lookup(self, timestamp: datetime) -> float:
        key = f"{timestamp.weekday()}-{timestamp.hour}"
        return float(self.seasonal_profile.get(key, self.global_average))

    def _zone_multiplier(self, junction_id: str) -> float:
        zone = self.junction_zones.get(junction_id, "standard")
        return self.zone_multipliers.get(zone, 1.0)

    def train_models(self, junction_id: str | None = None, force_retrain: bool = False):
        if self.model is not None and not force_retrain:
            metrics = dict(self.validation_metrics)
            metrics.update(
                {
                    "junction_id": junction_id,
                    "status": "loaded",
                    "model_path": str(self.model_path),
                }
            )
            return metrics

        table = self._build_training_table()
        if len(table) < 48:
            raise ValueError("Not enough hourly samples to train the forecaster.")

        split_idx = max(int(len(table) * 0.8), len(table) - 24 * 7)
        split_idx = min(split_idx, len(table) - 1)
        train_df = table.iloc[:split_idx].copy()
        val_df = table.iloc[split_idx:].copy()

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_depth=18,
            min_samples_leaf=2,
        )
        model.fit(train_df[self.feature_columns], train_df["count"])

        val_pred = np.clip(model.predict(val_df[self.feature_columns]), 0.0, None)
        y_true = val_df["count"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((y_true - val_pred) ** 2)))
        mape = float(np.mean(np.abs(y_true - val_pred) / np.maximum(y_true, 1.0)))
        residuals = y_true - val_pred

        seasonal_profile = (
            train_df.assign(day_key=train_df["dayofweek"].astype(int).astype(str) + "-" + train_df["hour"].astype(int).astype(str))
            .groupby("day_key")["count"]
            .mean()
            .to_dict()
        )

        self.model = model
        self.global_average = float(train_df["count"].mean())
        self.seasonal_profile = {str(k): float(v) for k, v in seasonal_profile.items()}
        self.residual_std = float(np.std(residuals))
        non_zero = train_df.loc[train_df["count"] > 0, "count"]
        threshold_source = non_zero if not non_zero.empty else train_df["count"]
        self.peak_threshold = max(1.0, float(np.percentile(threshold_source, 85)))
        self.validation_metrics = {
            "MAPE": mape,
            "RMSE": rmse,
            "samples": int(len(table)),
            "status": "trained",
        }

        artifact = {
            "model": self.model,
            "global_average": self.global_average,
            "seasonal_profile": self.seasonal_profile,
            "residual_std": self.residual_std,
            "peak_threshold": self.peak_threshold,
            "validation_metrics": self.validation_metrics,
            "trained_at": datetime.utcnow().isoformat(),
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, self.model_path)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.validation_metrics, f, indent=2)

        metrics = dict(self.validation_metrics)
        metrics.update(
            {
                "junction_id": junction_id,
                "status": "trained",
                "model_path": str(self.model_path),
            }
        )
        return metrics

    def _tree_prediction_std(self, feature_row: pd.DataFrame) -> float:
        if self.model is None or not hasattr(self.model, "estimators_"):
            return self.residual_std
        feature_array = feature_row.to_numpy()
        tree_preds = np.array([tree.predict(feature_array)[0] for tree in self.model.estimators_], dtype=float)
        return float(np.std(tree_preds))

    def predict(self, junction_id: str, hours: int = 24):
        if self.model is None:
            try:
                self.train_models(junction_id=junction_id)
            except Exception as exc:
                LOGGER.warning("Traffic forecaster is falling back to baseline mode: %s", exc)

        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        zone_multiplier = self._zone_multiplier(junction_id)
        forecasts = []

        for step in range(hours):
            ts = now + timedelta(hours=step)
            feature_row = self._build_features(pd.Series([ts]))[self.feature_columns]
            if self.model is not None:
                model_pred = float(self.model.predict(feature_row)[0])
                profile_pred = self._profile_lookup(ts)
                prediction = 0.7 * model_pred + 0.3 * profile_pred
                spread = max(self._tree_prediction_std(feature_row), self.residual_std)
            else:
                prediction = self._profile_lookup(ts)
                spread = max(self.residual_std, 1.0)

            prediction = max(0.0, prediction * zone_multiplier)
            interval = 1.96 * spread * zone_multiplier
            lower = max(0.0, prediction - interval)
            upper = prediction + interval

            forecasts.append(
                {
                    "timestamp": ts.isoformat(),
                    "predicted_violations": round(prediction, 2),
                    "confidence_interval": {
                        "lower": round(lower, 2),
                        "upper": round(upper, 2),
                    },
                    "event_flag": "PEAK" if prediction >= self.peak_threshold else None,
                }
            )

        return forecasts
