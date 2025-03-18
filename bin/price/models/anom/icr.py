import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from pyod.models.knn import KNN
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import joblib
import os
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IncrementalAnomalyModel(BaseEstimator):
    def __init__(self, stock: str, contamination=0.1, direction="both"):
        self.stock = stock
        if not 0 < contamination <= 1:
            raise ValueError(f"Contamination must be between 0 and 1, got {contamination}")
        self.contamination = contamination
        self.direction = direction
        self.scaler = RobustScaler()
        self.base_model = None
        self.meta_learner = SGDClassifier(loss='hinge', random_state=999, warm_start=True, class_weight='balanced')

    def _create_base_model(self):
        return KNN(contamination=self.contamination, n_neighbors=20)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X must be a DataFrame and y must be a Series")
        if X.empty or y.empty:
            raise ValueError("Input X or y is empty")
        self.feature_names = list(X.columns)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=999, stratify=y)
        X_scaled = self.scaler.fit_transform(X_train)
        self.base_model = self._create_base_model()
        self.base_model.fit(X_scaled)
        
        base_preds = self.base_model.decision_scores_.reshape(-1, 1)
        anom_score = base_preds[y_train == 1].mean()
        norm_score = base_preds[y_train == 0].mean()
        invert_scores = anom_score < norm_score
        if invert_scores:
            logging.warning("Anomalies have lower scores than normal data. Adjusting threshold.")
        
        base_preds_normalized = (base_preds - base_preds.min()) / (base_preds.max() - base_preds.min())
        self.meta_learner.fit(base_preds_normalized, y_train)
        
        X_val_scaled = self.scaler.transform(X_val)
        val_preds = self.base_model.decision_function(X_val_scaled).reshape(-1, 1)
        val_preds_normalized = (val_preds - val_preds.min()) / (val_preds.max() - val_preds.min())
        val_scores = self.meta_learner.decision_function(val_preds_normalized)
        self.threshold = np.percentile(val_scores, 100 * (self.contamination if invert_scores else (1 - self.contamination)))
        logging.debug(f"Validation threshold: {self.threshold}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        if not all(col in X.columns for col in self.feature_names):
            raise ValueError("Input X is missing some features present during training")
        X_scaled = self.scaler.transform(X[self.feature_names])
        base_preds = self.base_model.decision_function(X_scaled).reshape(-1, 1)
        base_preds_normalized = (base_preds - base_preds.min()) / (base_preds.max() - base_preds.min())
        scores = self.meta_learner.decision_function(base_preds_normalized)
        return pd.Series(np.where(scores > self.threshold, 1, 0), index=X.index, name='anomaly')

    def save_model(self, directory: str = "bin/anom/models/saved"):
        filepath = os.path.join(directory, f"anomaly_model_{self.stock}_{self.direction}.pkl")
        os.makedirs(directory, exist_ok=True)
        joblib.dump({
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'base_model': self.base_model,
            'meta_learner': self.meta_learner,
            'threshold': self.threshold,
            'stock': self.stock,
            'direction': self.direction
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load_model(self, directory: str = "bin/anom/models/saved"):
        filepath = os.path.join(directory, f"anomaly_model_{self.stock}_{self.direction}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        data = joblib.load(filepath)
        self.feature_names = data['feature_names']
        self.scaler = data['scaler']
        self.base_model = data['base_model']
        self.meta_learner = data['meta_learner']
        self.threshold = data['threshold']
        self.stock = data['stock']
        self.direction = data['direction']

sys.path.append(str(Path(__file__).resolve().parents[3]))
from bin.models.option_stats_model_setup import data
from bin.main import get_path

class AnomalyScanner:
    def __init__(self, connections):
        if connections is None:
            raise ValueError("Connections parameter cannot be None")
        self.data_connector = data(connections=connections)

    def _select_features(self, X, y_raw, direction):
        """Select features using correlation and mutual information with statistical significance."""
        if direction == "bullish":
            y = (y_raw > 0.01).astype(int)
        elif direction == "bearish":
            y = (y_raw < -0.01).astype(int)
        else:  # both
            y = (np.abs(y_raw) > 0.01).astype(int)
        
        # Compute Pearson correlation with p-values
        corr_dict = {}
        for col in X.columns:
            corr, p_val = pearsonr(X[col], y)
            corr_dict[col] = {'corr': corr, 'p_val': p_val}
        corr_df = pd.DataFrame(corr_dict).T
        
        # Compute mutual information
        mi_scores = mutual_info_classif(X, y, random_state=999)
        mi_df = pd.DataFrame({'mi': mi_scores}, index=X.columns)
        
        # Combine scores
        feature_scores = corr_df.join(mi_df)
        feature_scores['combined_score'] = 0.7 * feature_scores['corr'].abs() + 0.3 * feature_scores['mi']  # Weighted combination
        
        # Filter by direction, significance, and thresholds
        if direction == "bullish":
            selected = feature_scores[(feature_scores['corr'] > 0.01) & 
                                     (feature_scores['p_val'] < 0.30) & 
                                     (feature_scores['mi'] > 0.01)]
        elif direction == "bearish":
            selected = feature_scores[(feature_scores['corr'] < 0) & 
                                     (feature_scores['p_val'] < 1) & 
                                     (feature_scores['mi'] > 0.01)]
        else:  # both
            selected = feature_scores[(abs(feature_scores['corr']) > 0.01) & 
                                     (feature_scores['p_val'] < 0.30) & 
                                     (feature_scores['mi'] > 0.01)]
        
        selected_features = selected.index.tolist()
        
        # Remove highly correlated features (redundancy filter)
        if selected_features:
            corr_matrix = X[selected_features].corr()
            to_drop = set()
            for i in range(len(selected_features)):
                for j in range(i + 1, len(selected_features)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        # Drop the feature with lower combined score
                        if selected.loc[selected_features[i], 'combined_score'] > selected.loc[selected_features[j], 'combined_score']:
                            to_drop.add(selected_features[j])
                        else:
                            to_drop.add(selected_features[i])
            selected_features = [f for f in selected_features if f not in to_drop]
        
        if not selected_features:
            logging.warning(f"\n\nNo features selected for {direction}. Using all features.\n\n")
            selected_features = X.columns.tolist()
        
        logging.debug(f"Feature scores for {direction}:\n{feature_scores}")
        logging.debug(f"Selected features for {direction}: {selected_features}")
        return X[selected_features]

    def _stock_data(self, stock, start_date=None, end_date=None, anomaly_threshold=0.01, direction="both"):
        if not isinstance(stock, str):
            raise TypeError("Stock must be a string")
        try:
            x, y_raw = self.data_connector._returnxy(stock, start_date=start_date, end_date=end_date)
            if x.empty or y_raw.empty:
                raise ValueError(f"No data returned for stock {stock} between {start_date} and {end_date}")
            
            if direction not in ["both", "bullish", "bearish"]:
                raise ValueError("Direction must be 'both', 'bullish', or 'bearish'")
            
            if direction == "both":
                y = pd.Series(np.where(np.abs(y_raw) > anomaly_threshold, 1, 0), index=y_raw.index, name='target')
            elif direction == "bullish":
                y = pd.Series(np.where(y_raw > anomaly_threshold, 1, 0), index=y_raw.index, name='target')
            elif direction == "bearish":
                y = pd.Series(np.where(y_raw < -anomaly_threshold, 1, 0), index=y_raw.index, name='target')
            
            x = x.loc[y.index]
            if x.empty or y.empty:
                raise ValueError(f"No data after filtering for anomalies/features for stock {stock}")
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Mismatch in rows between x and y for {stock}: {x.shape[0]} vs {y.shape[0]}")
            
            logging.debug(f"Loaded data for {stock} ({direction}): X {x.shape}, Y {y.shape}, Features: {x.columns.tolist()}")
            return x, y
        except Exception as e:
            logging.error(f"Failed to load stock data for {stock}: {str(e)}")
            raise

    def log_counts(self, preds, y, phase='Training'):
        predicted_count = preds.value_counts()
        actual_count = y.value_counts()
        logging.info(f"{phase} Anomaly Counts - Predicted {phase}: {predicted_count.get(1, 0)} - Actual {phase}: {actual_count.get(1, 0)}")

    def incremental_model(self, stock, contamination=0.1, direction="both", **kwargs):
        x, y = self._stock_data(stock, direction=direction, **kwargs)
        x = self._select_features(x, y, direction=direction)
        model = IncrementalAnomalyModel(stock=stock, contamination=contamination, direction=direction)
        model.fit(x, y)
        preds = model.predict(x)
        self.log_counts(preds, y, phase='Training')
        self.save_model(model)
        return model

    def save_model(self, model, directory="bin/models/saved"):
        try:
            model.save_model(directory)
            logging.debug(f"Model saved for {model.stock} ({model.direction}) to {directory}")
        except Exception as e:
            logging.error(f"Failed to save model for {model.stock}: {str(e)}")
            raise

    def load_model(self, stock, direction, directory="bin/models/saved"):
        try:
            model = IncrementalAnomalyModel(stock=stock, direction=direction)
            model.load_model(directory)
            logging.debug(f"Model loaded for {stock} ({direction}) from {directory}")
            return model
        except FileNotFoundError as e:
            logging.error(f"Failed to load model for {stock} ({direction}): {str(e)}")
            raise

    def __track_n_day_returns(self, stock, preds, X, n_days=5):
        signal_dates = X.index[preds == 1]
        price_data = self.data_connector.price_data(stock)
        returns = []
        for date in signal_dates:
            if date in price_data.index:
                idx = price_data.index.get_loc(date)
                if idx + n_days < len(price_data):
                    start_price = price_data.iloc[idx]['close']
                    end_price = price_data.iloc[idx + n_days]['close']
                    ret = (end_price / start_price - 1) * 100
                    returns.append(ret)
        return returns

    def run(self, stock, contamination=0.1, start_date=None, end_date=None, anomaly_threshold=0.01, direction="both", n_days=5):
        logging.info(f"${stock.upper()} contamination: {contamination:.2f}, checking for {direction} anomalies")

        # Fit and Save the model 
        model_fit = self.incremental_model(stock, contamination, direction=direction, start_date=start_date, 
                                          end_date=end_date, anomaly_threshold=anomaly_threshold)
        # Load the Saved Model 
        model = self.load_model(stock, direction)
        xtest, ytest = self._stock_data(stock, start_date="2025-01-01", anomaly_threshold=anomaly_threshold, direction=direction)
        logging.debug(f'X_test: {xtest.shape}, Y_test: {ytest.shape}')
        
        preds_test = model.predict(xtest)
        self.log_counts(preds_test, ytest, phase='Testing')
        
        returns = self.__track_n_day_returns(stock, preds_test, xtest, n_days=n_days)
        if returns:
            avg_ret = np.mean(returns)
            max_ret = np.max(returns)
            min_ret = np.min(returns)
            std_ret = np.std(returns)
            logging.info(f"Average {n_days}D returns for detected anomalies: {avg_ret:.2f}%, max: {max_ret:.2f}%, min: {min_ret:.2f}%, std: {std_ret:.2f}%")

        return model

if __name__ == "__main__":
    connections = get_path()
    scanner = AnomalyScanner(connections)
    stocks = ['spy', 'mo', 'iwm']
    directions = ['bullish', 'bearish', 'both']
    for stock in stocks:
        for direction in directions:
            for contamination in [0.05, 0.10]:
                scanner.run(stock, contamination=contamination, end_date="2025-01-01", direction=direction, n_days=5)
            print('\n')
        print('\n\n')