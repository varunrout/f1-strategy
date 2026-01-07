"""
Tyre Degradation Model

XGBoost-based model for predicting F1 tyre degradation rates.
Includes baseline comparisons, quantile predictions, and SHAP explanations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

# ML imports
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn or xgboost not installed. Install with: pip install xgboost scikit-learn")


class DegradationModel:
    """XGBoost model for tyre degradation prediction."""
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize degradation model.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.quantile_models = {}
        self.scaler = StandardScaler()
        self.baseline_models = {}
        self.feature_columns = []
        self.metrics = {}
        
    def train_baselines(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Train baseline models for comparison.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dict with baseline metrics
        """
        results = {}
        
        # Baseline 1: Circuit mean
        if 'circuit_baseline_deg' in X_train.columns:
            pred_circuit = X_val['circuit_baseline_deg']
            results['circuit_mean'] = {
                'mae': mean_absolute_error(y_val, pred_circuit),
                'rmse': np.sqrt(mean_squared_error(y_val, pred_circuit)),
                'r2': r2_score(y_val, pred_circuit)
            }
        
        # Baseline 2: Driver baseline
        if 'driver_baseline_deg' in X_train.columns:
            pred_driver = X_val['driver_baseline_deg']
            results['driver_baseline'] = {
                'mae': mean_absolute_error(y_val, pred_driver),
                'rmse': np.sqrt(mean_squared_error(y_val, pred_driver)),
                'r2': r2_score(y_val, pred_driver)
            }
        
        # Baseline 3: Linear regression on baselines only
        baseline_cols = [c for c in X_train.columns if 'baseline' in c]
        if baseline_cols:
            lr_baseline = LinearRegression()
            lr_baseline.fit(X_train[baseline_cols], y_train)
            pred_lr = lr_baseline.predict(X_val[baseline_cols])
            results['linear_baselines'] = {
                'mae': mean_absolute_error(y_val, pred_lr),
                'rmse': np.sqrt(mean_squared_error(y_val, pred_lr)),
                'r2': r2_score(y_val, pred_lr)
            }
            self.baseline_models['linear_baselines'] = lr_baseline
        
        # Baseline 4: Linear regression with teammate delta
        if 'teammate_delta' in X_train.columns:
            delta_cols = baseline_cols + ['teammate_delta']
            lr_delta = LinearRegression()
            lr_delta.fit(X_train[delta_cols], y_train)
            pred_delta = lr_delta.predict(X_val[delta_cols])
            results['linear_with_teammate'] = {
                'mae': mean_absolute_error(y_val, pred_delta),
                'rmse': np.sqrt(mean_squared_error(y_val, pred_delta)),
                'r2': r2_score(y_val, pred_delta)
            }
            self.baseline_models['linear_with_teammate'] = lr_delta
        
        self.metrics['baselines'] = results
        return results
    
    def train(self, df: pd.DataFrame, feature_columns: List[str], 
              target_column: str = 'deg_rate_per_lap',
              test_year: int = 2024,
              params: Optional[Dict] = None) -> Dict:
        """
        Train XGBoost model with train/val/test split.
        
        Args:
            df: Full DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Target column name
            test_year: Year to use as holdout test set
            params: Optional XGBoost parameters
            
        Returns:
            Dict with training metrics
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn and xgboost required for training")
        
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Split data
        # Train: 2022-2023, Val: early 2024, Test: late 2024
        train_mask = df['year'] < test_year
        test_mask = df['year'] == test_year
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        # Further split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            train_df[feature_columns], 
            train_df[target_column],
            test_size=0.2,
            random_state=42
        )
        
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]
        
        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Train baselines
        print("\nTraining baseline models...")
        baseline_results = self.train_baselines(X_train, y_train, X_val, y_val)
        
        # Default XGBoost parameters
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50
        }
        
        if params:
            default_params.update(params)
        
        # Train XGBoost
        print("\nTraining XGBoost model...")
        self.model = xgb.XGBRegressor(**default_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation
        y_pred_val = self.model.predict(X_val)
        val_metrics = {
            'mae': mean_absolute_error(y_val, y_pred_val),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'r2': r2_score(y_val, y_pred_val)
        }
        
        # Evaluate on test (2024 holdout)
        y_pred_test = self.model.predict(X_test)
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
        
        self.metrics['validation'] = val_metrics
        self.metrics['test'] = test_metrics
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.metrics['feature_importance'] = importance.to_dict('records')
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETE")
        print("=" * 60)
        
        print("\nBaseline Results (Validation):")
        for name, metrics in baseline_results.items():
            print(f"  {name}: MAE={metrics['mae']*1000:.1f} ms/lap, R²={metrics['r2']:.3f}")
        
        print(f"\nXGBoost Validation: MAE={val_metrics['mae']*1000:.1f} ms/lap, R²={val_metrics['r2']:.3f}")
        print(f"XGBoost Test (2024): MAE={test_metrics['mae']*1000:.1f} ms/lap, R²={test_metrics['r2']:.3f}")
        
        print("\nTop 5 Features:")
        for i, row in importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'baselines': baseline_results,
            'validation': val_metrics,
            'test': test_metrics,
            'feature_importance': importance
        }
    
    def train_quantile_models(self, df: pd.DataFrame, feature_columns: List[str],
                               target_column: str = 'deg_rate_per_lap',
                               quantiles: List[float] = [0.1, 0.5, 0.9]) -> Dict:
        """
        Train quantile regression models for uncertainty estimation.
        
        Args:
            df: Full DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Target column name
            quantiles: List of quantiles to predict
            
        Returns:
            Dict with quantile model metrics
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn and xgboost required for training")
        
        # Use 2022-2023 for training
        train_df = df[df['year'] < 2024]
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        
        results = {}
        
        for q in quantiles:
            print(f"Training quantile model for P{int(q*100)}...")
            
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train, verbose=False)
            self.quantile_models[q] = model
            
            results[f'p{int(q*100)}'] = {'quantile': q, 'trained': True}
        
        print(f"✅ Trained {len(quantiles)} quantile models")
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict degradation rate.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X[self.feature_columns])
    
    def predict_with_intervals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict degradation rate with uncertainty intervals.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with P10, P50, P90 predictions
        """
        if not self.quantile_models:
            raise ValueError("Quantile models not trained. Call train_quantile_models() first.")
        
        results = pd.DataFrame(index=X.index)
        
        for q, model in self.quantile_models.items():
            results[f'deg_p{int(q*100)}'] = model.predict(X[self.feature_columns])
        
        # Also add point prediction from main model
        if self.model is not None:
            results['deg_point'] = self.model.predict(X[self.feature_columns])
        
        return results
    
    def save(self, name: str = "deg_model"):
        """
        Save model to disk.
        
        Args:
            name: Model name for files
        """
        import joblib
        
        # Save main model using joblib for sklearn compatibility
        if self.model is not None:
            joblib.dump(self.model, self.model_dir / f"{name}_main.joblib")
        
        # Save quantile models using joblib
        for q, model in self.quantile_models.items():
            joblib.dump(model, self.model_dir / f"{name}_q{int(q*100)}.joblib")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'target_column': getattr(self, 'target_column', 'deg_rate_per_lap'),
            'metrics': self.metrics
        }
        
        with open(self.model_dir / f"{name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Model saved to {self.model_dir}")
    
    def load(self, name: str = "deg_model"):
        """
        Load model from disk.
        
        Args:
            name: Model name to load
        """
        import joblib
        
        # Load main model
        main_path = self.model_dir / f"{name}_main.joblib"
        if main_path.exists():
            self.model = joblib.load(main_path)
        
        # Load quantile models
        for q in [0.1, 0.5, 0.9]:
            q_path = self.model_dir / f"{name}_q{int(q*100)}.joblib"
            if q_path.exists():
                self.quantile_models[q] = joblib.load(q_path)
        
        # Load metadata
        meta_path = self.model_dir / f"{name}_metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            self.feature_columns = metadata.get('feature_columns', [])
            self.target_column = metadata.get('target_column', 'deg_rate_per_lap')
            self.metrics = metadata.get('metrics', {})
        
        print(f"✅ Model loaded from {self.model_dir}")
    
    def get_feature_importance_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get SHAP-based feature importance.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with SHAP importance
        """
        try:
            import shap
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return pd.DataFrame()
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X[self.feature_columns])
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        return importance
    
    def evaluate_by_segment(self, df: pd.DataFrame, 
                            segment_col: str) -> pd.DataFrame:
        """
        Evaluate model performance by segment (e.g., circuit, compound).
        
        Args:
            df: DataFrame with features, target, and segment column
            segment_col: Column to segment by
            
        Returns:
            DataFrame with metrics per segment
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        results = []
        
        for segment in df[segment_col].unique():
            mask = df[segment_col] == segment
            X_seg = df.loc[mask, self.feature_columns]
            y_seg = df.loc[mask, self.target_column]
            
            if len(y_seg) < 5:
                continue
            
            y_pred = self.model.predict(X_seg)
            
            results.append({
                segment_col: segment,
                'n_stints': len(y_seg),
                'mae': mean_absolute_error(y_seg, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_seg, y_pred)),
                'r2': r2_score(y_seg, y_pred) if len(y_seg) > 1 else np.nan
            })
        
        return pd.DataFrame(results).sort_values('mae')


class StrategyPredictor:
    """High-level strategy predictions using degradation model."""
    
    def __init__(self, model: DegradationModel):
        """
        Initialize strategy predictor.
        
        Args:
            model: Trained DegradationModel
        """
        self.model = model
    
    def predict_stint_degradation(self, driver: str, team: str, circuit: str,
                                   compound: str, stint_number: int,
                                   track_temp: float = 35.0,
                                   grid_position: int = 10) -> Dict:
        """
        Predict degradation for a specific stint scenario.
        
        Args:
            driver: Driver code (e.g., 'VER')
            team: Team name
            circuit: GP name
            compound: Tyre compound
            stint_number: Stint number (1, 2, 3, ...)
            track_temp: Track temperature (°C)
            grid_position: Starting position
            
        Returns:
            Dict with predictions and intervals
        """
        # This would use the model's baselines and cluster mappings
        # For now, return placeholder
        return {
            'driver': driver,
            'compound': compound,
            'predicted_deg_rate': 0.025,  # ms/lap
            'p10_deg_rate': 0.015,
            'p90_deg_rate': 0.040,
            'expected_usable_laps': 25,
            'confidence': 'medium'
        }
    
    def simulate_strategy(self, driver: str, team: str, circuit: str,
                          race_laps: int, available_compounds: List[str],
                          track_temp: float = 35.0) -> pd.DataFrame:
        """
        Simulate different pit strategies.
        
        Args:
            driver: Driver code
            team: Team name
            circuit: GP name
            race_laps: Total race laps
            available_compounds: List of compounds for the race
            track_temp: Track temperature
            
        Returns:
            DataFrame with strategy options and expected times
        """
        # Placeholder for strategy simulation
        strategies = [
            {'strategy': 'One-stop (M-H)', 'expected_time': '+0.0s', 'risk': 'low'},
            {'strategy': 'One-stop (H-M)', 'expected_time': '+2.5s', 'risk': 'low'},
            {'strategy': 'Two-stop (S-M-H)', 'expected_time': '+1.2s', 'risk': 'medium'},
        ]
        return pd.DataFrame(strategies)


if __name__ == "__main__":
    # Quick test
    from feature_builder import FeatureBuilder
    
    print("Building features...")
    builder = FeatureBuilder()
    df = builder.build_full_feature_set()
    
    print("\nTraining model...")
    model = DegradationModel()
    results = model.train(
        df, 
        feature_columns=builder.get_feature_columns(),
        target_column=builder.get_target_column()
    )
    
    print("\nTraining quantile models...")
    model.train_quantile_models(
        df,
        feature_columns=builder.get_feature_columns()
    )
    
    print("\nSaving model...")
    model.save("deg_model_v1")
    
    print("\n✅ Model training complete!")
