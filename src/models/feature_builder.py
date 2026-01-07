"""
Feature Builder for Tyre Degradation Modeling

Builds the complete feature set for degradation prediction including:
- Historical baselines (circuit, team, driver, compound)
- Cluster-derived features (from domain1 cluster analysis)
- Relational features (teammate delta)
- Stint context features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import re
import unicodedata
import warnings

warnings.filterwarnings('ignore')


class FeatureBuilder:
    """Builds features for tyre degradation modeling."""
    
    def __init__(self, silver_path: str = "data/lake/silver/domain1"):
        """
        Initialize feature builder.
        
        Args:
            silver_path: Path to silver layer parquet files
        """
        self.silver_path = Path(silver_path)
        self.baselines = {}
        self.cluster_labels = {}

    @staticmethod
    def _normalize_gp_name(value: object) -> str:
        """Normalize GP name into a stable underscore key for rule-based mappings."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = str(value).strip()
        # Strip accents/diacritics (e.g., São -> Sao)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        # Normalize separators/spaces
        text = text.replace("-", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" ", "_")
        text = re.sub(r"_+", "_", text)
        return text

    @staticmethod
    def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None
        
    def load_base_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load base stint and lap data from silver layer."""
        stints_path = self.silver_path / "stints_degradation.parquet"
        laps_path = self.silver_path / "clean_air_laps.parquet"
        
        stints = pd.read_parquet(stints_path)
        laps = pd.read_parquet(laps_path)
        
        # Create stint key for joining
        stints['stint_key'] = (
            stints['year'].astype(str) + '_' +
            stints['gp_name'].astype(str) + '_' +
            stints['driver'].astype(str) + '_' +
            stints['stint'].astype(str)
        )
        
        # Filter to high-quality fits
        stints = stints[
            (stints['deg_r_squared'] > 0.3) & 
            (stints['deg_rate_per_lap'].abs() < 0.5) &
            (stints['num_laps'] >= 6)
        ].copy()
        
        print(f"Loaded {len(stints)} high-quality stints")
        return stints, laps
    
    def compute_historical_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute historical baseline degradation rates.
        
        Args:
            df: DataFrame with stint data
            
        Returns:
            DataFrame with baseline features added
        """
        # Circuit baseline (leave-one-out to prevent leakage)
        circuit_baselines = df.groupby('gp_name')['deg_rate_per_lap'].transform(
            lambda x: (x.sum() - x) / (len(x) - 1) if len(x) > 1 else np.nan
        )
        df['circuit_baseline_deg'] = circuit_baselines.fillna(
            df.groupby('gp_name')['deg_rate_per_lap'].transform('mean')
        )
        
        # Team baseline
        team_baselines = df.groupby('team')['deg_rate_per_lap'].transform(
            lambda x: (x.sum() - x) / (len(x) - 1) if len(x) > 1 else np.nan
        )
        df['team_baseline_deg'] = team_baselines.fillna(
            df.groupby('team')['deg_rate_per_lap'].transform('mean')
        )
        
        # Driver baseline
        driver_baselines = df.groupby('driver')['deg_rate_per_lap'].transform(
            lambda x: (x.sum() - x) / (len(x) - 1) if len(x) > 1 else np.nan
        )
        df['driver_baseline_deg'] = driver_baselines.fillna(
            df.groupby('driver')['deg_rate_per_lap'].transform('mean')
        )
        
        # Compound × Circuit baseline
        compound_circuit = df.groupby(['gp_name', 'compound'])['deg_rate_per_lap'].transform(
            lambda x: (x.sum() - x) / (len(x) - 1) if len(x) > 1 else np.nan
        )
        df['compound_circuit_baseline'] = compound_circuit.fillna(
            df.groupby(['gp_name', 'compound'])['deg_rate_per_lap'].transform('mean')
        )
        
        # Store global baselines for inference
        self.baselines = {
            'circuit': df.groupby('gp_name')['deg_rate_per_lap'].mean().to_dict(),
            'team': df.groupby('team')['deg_rate_per_lap'].mean().to_dict(),
            'driver': df.groupby('driver')['deg_rate_per_lap'].mean().to_dict(),
            'compound_circuit': df.groupby(['gp_name', 'compound'])['deg_rate_per_lap'].mean().to_dict()
        }
        
        print(f"Computed baselines: {len(self.baselines['circuit'])} circuits, "
              f"{len(self.baselines['team'])} teams, {len(self.baselines['driver'])} drivers")
        
        return df
    
    def compute_teammate_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute teammate delta - driver deg relative to teammate in same car.
        
        This is the breakthrough feature that captures driver technique
        independent of car/setup effects.
        
        Args:
            df: DataFrame with stint data
            
        Returns:
            DataFrame with teammate delta features
        """
        # Group by year, GP, team to find teammates
        df_copy = df.copy()
        
        teammate_stats = df_copy.groupby(['year', 'gp_name', 'team']).agg({
            'deg_rate_per_lap': ['mean', 'count']
        }).reset_index()
        teammate_stats.columns = ['year', 'gp_name', 'team', 'team_race_avg', 'team_race_count']
        
        # Join back
        df = df.merge(teammate_stats, on=['year', 'gp_name', 'team'], how='left')
        
        # Compute teammate delta
        df['teammate_delta'] = df['deg_rate_per_lap'] - df['team_race_avg']
        
        # Compute teammate average (different from team_race_avg when driver is excluded)
        # For single-driver teams, teammate_delta = 0
        df['teammate_delta'] = df['teammate_delta'].fillna(0)
        
        print(f"Computed teammate delta for {len(df)} stints")
        print(f"Teammate delta range: [{df['teammate_delta'].min():.4f}, {df['teammate_delta'].max():.4f}]")
        
        return df
    
    def add_circuit_clusters(self, df: pd.DataFrame, 
                              cluster_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """
        Add circuit cluster labels from domain1 cluster analysis.
        
        Args:
            df: DataFrame with stint data
            cluster_mapping: Dict mapping gp_name to cluster ID
            
        Returns:
            DataFrame with circuit cluster feature
        """
        gp_key = df['gp_name'].apply(self._normalize_gp_name)

        if cluster_mapping is None:
            # Default clusters based on degradation characteristics (from analysis)
            # C0: High-deg circuits (Spa, Bahrain, Spain)
            # C1: Medium-deg circuits (most tracks)
            # C2: Low-deg/recovery circuits (Monaco, Singapore)
            # C3: Variable circuits (depends on conditions)
            high_deg = ['Belgian_Grand_Prix', 'Bahrain_Grand_Prix', 'Spanish_Grand_Prix', 
                       'Hungarian_Grand_Prix', 'Japanese_Grand_Prix']
            low_deg = ['Monaco_Grand_Prix', 'Singapore_Grand_Prix', 'Azerbaijan_Grand_Prix',
                      'Australian_Grand_Prix']
            
            def assign_circuit_cluster(gp):
                gp = self._normalize_gp_name(gp)
                if gp in high_deg:
                    return 0  # High-deg
                elif gp in low_deg:
                    return 2  # Low-deg
                else:
                    return 1  # Medium
            
            df['circuit_cluster'] = gp_key.apply(assign_circuit_cluster).astype(int)
        else:
            normalized_map = {self._normalize_gp_name(k): v for k, v in cluster_mapping.items()}
            df['circuit_cluster'] = gp_key.map(normalized_map).fillna(-1).astype(int)
        
        return df
    
    def add_team_clusters(self, df: pd.DataFrame,
                           cluster_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """
        Add team cluster labels (strategy philosophy).
        
        Args:
            df: DataFrame with stint data
            cluster_mapping: Dict mapping team to cluster ID
            
        Returns:
            DataFrame with team cluster feature
        """
        if cluster_mapping is None:
            # Default clusters based on team strategy DNA (from analysis)
            # C0: Conservative (long stints, low deg)
            # C1: Balanced
            # C2: Aggressive (multi-stop, high deg tolerance)
            conservative = ['Mercedes', 'Alpine', 'RB']
            aggressive = ['Ferrari', 'Williams', 'Alfa Romeo', 'Kick Sauber']
            
            def assign_team_cluster(team):
                if team in conservative:
                    return 0
                elif team in aggressive:
                    return 2
                else:
                    return 1
            
            df['team_cluster'] = df['team'].apply(assign_team_cluster)
        else:
            df['team_cluster'] = df['team'].map(cluster_mapping).fillna(-1).astype(int)
        
        return df
    
    def add_driver_clusters(self, df: pd.DataFrame,
                             cluster_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """
        Add driver cluster labels (tyre management style).
        
        Args:
            df: DataFrame with stint data
            cluster_mapping: Dict mapping driver to cluster ID
            
        Returns:
            DataFrame with driver cluster feature
        """
        if cluster_mapping is None:
            # Default clusters based on driver tyre management (from analysis)
            # C0: Excellent tyre managers (low deg)
            # C1: Good managers
            # C2: Average
            # C3: Higher deg drivers
            excellent = ['TSU', 'VET', 'GAS', 'LEC']
            good = ['VER', 'HAM', 'NOR', 'PIA']
            poor = ['LAT', 'DEV', 'ALB', 'MAG']
            
            def assign_driver_cluster(driver):
                if driver in excellent:
                    return 0
                elif driver in good:
                    return 1
                elif driver in poor:
                    return 3
                else:
                    return 2
            
            df['driver_cluster'] = df['driver'].apply(assign_driver_cluster)
        else:
            df['driver_cluster'] = df['driver'].map(cluster_mapping).fillna(-1).astype(int)
        
        return df
    
    def add_stint_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add stint context features.
        
        Args:
            df: DataFrame with stint data
            
        Returns:
            DataFrame with context features
        """
        # Stint number (1, 2, 3, etc.)
        # Already in data as 'stint'
        
        # Position tier
        position_col = self._first_existing_col(
            df,
            [
                'position',
                'avg_position',
                'best_position',
                'worst_position',
                'finish_position',
                'grid_position',
            ],
        )
        if position_col is not None:
            pos = pd.to_numeric(df[position_col], errors='coerce')
            df['position_tier'] = pd.cut(
                pos,
                bins=[0, 6, 14, 60],
                labels=[0, 1, 2],  # Front, Mid, Back
                include_lowest=True,
            ).astype(float).fillna(1)
        else:
            df['position_tier'] = 1  # Default to mid
        
        # Compound encoding
        compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
        df['compound_encoded'] = df['compound'].map(compound_map).fillna(1)
        
        # Downforce class based on circuit
        high_df_circuits = ['Monaco_Grand_Prix', 'Singapore_Grand_Prix', 'Hungarian_Grand_Prix']
        low_df_circuits = ['Italian_Grand_Prix', 'Belgian_Grand_Prix', 'Saudi_Arabian_Grand_Prix',
                          'Azerbaijan_Grand_Prix', 'Las_Vegas_Grand_Prix']
        
        def get_downforce(gp):
            gp = self._normalize_gp_name(gp)
            if gp in high_df_circuits:
                return 2  # High
            elif gp in low_df_circuits:
                return 0  # Low
            else:
                return 1  # Medium
        
        df['downforce_class'] = df['gp_name'].apply(get_downforce).astype(int)
        
        return df
    
    def add_shape_cluster(self, df: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
        """
        Add stint shape cluster based on degradation curve pattern.
        
        Args:
            df: DataFrame with stint data
            laps: DataFrame with lap-level data
            
        Returns:
            DataFrame with shape cluster feature
        """
        # Compute shape features per stint
        shape_features = []
        
        for stint_key in df['stint_key'].unique():
            stint_laps = laps[laps['stint_key'] == stint_key].sort_values('lap_number')
            
            if len(stint_laps) < 6:
                shape_features.append({'stint_key': stint_key, 'shape_cluster': -1})
                continue
            
            # Compute slope in thirds
            n = len(stint_laps)
            third = n // 3
            
            if 'lap_time_s' not in stint_laps.columns:
                shape_features.append({'stint_key': stint_key, 'shape_cluster': -1})
                continue
                
            times = stint_laps['lap_time_s'].values
            tyre_life = stint_laps['tyre_life'].values
            
            # Early slope
            early_coef = np.polyfit(tyre_life[:third], times[:third], 1) if third >= 2 else [0, 0]
            # Late slope
            late_coef = np.polyfit(tyre_life[-third:], times[-third:], 1) if third >= 2 else [0, 0]
            
            slope_early = early_coef[0]
            slope_late = late_coef[0]
            curvature = slope_late - slope_early
            
            # Classify shape
            # C0: Linear (similar slopes)
            # C1: Progressive (accelerating deg)
            # C2: Cliff (late spike)
            # C3: Recovery (negative slope)
            
            if abs(curvature) < 0.02:
                if slope_early < -0.02:
                    shape = 3  # Recovery
                else:
                    shape = 0  # Linear
            elif curvature > 0.05:
                shape = 1  # Progressive
            elif slope_late > 0.1:
                shape = 2  # Cliff
            else:
                shape = 0  # Linear
            
            shape_features.append({'stint_key': stint_key, 'shape_cluster': shape})
        
        shape_df = pd.DataFrame(shape_features)
        df = df.merge(shape_df, on='stint_key', how='left')
        df['shape_cluster'] = df['shape_cluster'].fillna(-1).astype(int)
        
        return df
    
    def build_full_feature_set(self, 
                                cluster_mappings: Optional[Dict] = None) -> pd.DataFrame:
        """
        Build complete feature set for modeling.
        
        Args:
            cluster_mappings: Optional dict with cluster label mappings from analysis
            
        Returns:
            DataFrame ready for modeling with all features
        """
        print("=" * 60)
        print("BUILDING FEATURE SET FOR TYRE DEGRADATION MODELING")
        print("=" * 60)
        
        # Load base data
        stints, laps = self.load_base_data()
        
        # Create stint keys in laps too
        if 'stint_key' not in laps.columns:
            laps['stint_key'] = (
                laps['year'].astype(str) + '_' +
                laps['gp_name'].astype(str) + '_' +
                laps['driver'].astype(str) + '_' +
                laps['stint'].astype(str)
            )
        
        # Compute historical baselines
        print("\n1. Computing historical baselines...")
        stints = self.compute_historical_baselines(stints)
        
        # Compute teammate delta (THE BREAKTHROUGH FEATURE)
        print("\n2. Computing teammate delta...")
        stints = self.compute_teammate_delta(stints)
        
        # Add cluster labels
        print("\n3. Adding cluster labels...")
        cluster_map = cluster_mappings or {}
        stints = self.add_circuit_clusters(stints, cluster_map.get('circuit'))
        stints = self.add_team_clusters(stints, cluster_map.get('team'))
        stints = self.add_driver_clusters(stints, cluster_map.get('driver'))
        
        # Add shape cluster (requires lap data)
        print("\n4. Computing shape clusters...")
        stints = self.add_shape_cluster(stints, laps)
        
        # Add stint context
        print("\n5. Adding stint context features...")
        stints = self.add_stint_context(stints)
        
        # Final feature list
        features = [
            # Target
            'deg_rate_per_lap',
            
            # Historical baselines
            'circuit_baseline_deg', 'team_baseline_deg', 'driver_baseline_deg',
            'compound_circuit_baseline',
            
            # Relational (BREAKTHROUGH)
            'teammate_delta',
            
            # Cluster labels
            'circuit_cluster', 'team_cluster', 'driver_cluster', 'shape_cluster',
            
            # Context
            'stint', 'num_laps', 'compound_encoded', 'downforce_class', 'position_tier',
            
            # Metadata (for analysis, not training)
            'stint_key', 'year', 'gp_name', 'driver', 'team', 'compound', 'deg_r_squared'
        ]
        
        # Keep only available features
        available = [f for f in features if f in stints.columns]
        df = stints[available].copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        print(f"\n✅ Feature set built: {len(df)} stints × {len(available)} features")
        print(f"\nFeatures: {[f for f in available if f not in ['stint_key', 'year', 'gp_name', 'driver', 'team', 'compound', 'deg_r_squared']]}")
        
        return df
    
    def get_feature_columns(self) -> list:
        """Return list of feature columns for training (excludes target and metadata)."""
        return [
            # Historical baselines
            'circuit_baseline_deg', 'team_baseline_deg', 'driver_baseline_deg',
            'compound_circuit_baseline',
            
            # Relational
            'teammate_delta',
            
            # Cluster labels
            'circuit_cluster', 'team_cluster', 'driver_cluster', 'shape_cluster',
            
            # Context
            'stint', 'num_laps', 'compound_encoded', 'downforce_class', 'position_tier'
        ]
    
    def get_target_column(self) -> str:
        """Return target column name."""
        return 'deg_rate_per_lap'


if __name__ == "__main__":
    # Test feature builder
    builder = FeatureBuilder()
    df = builder.build_full_feature_set()
    print(df.head())
    print(df.describe())
