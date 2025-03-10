"""
Anomaly detection model using various algorithms
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats 
from logging import getLogger, Logger, DEBUG, INFO, Formatter, StreamHandler
import logging
from typing import Union, Optional, Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import TimeSeriesSplit
from numpy.typing import NDArray

import sys
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from connect import setup

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning,
                     message='X does not have valid feature names')

# Configure module logger
logger = getLogger(__name__)
logger.propagate = False  # Prevent duplicate logging
if not logger.handlers:
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ModelType = Union[IsolationForest, OneClassSVM, LocalOutlierFactor, KMeans]
PredictionType = Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
DecompType = Union[PCA, KernelPCA, FastICA]

class anomaly_model(setup):
    """Anomaly detection model using various algorithms"""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        feature_names: List[str], 
        target_names: List[str],
        stock: str,
        verbose: bool = False, 
        **kwargs
    ) -> None:
        """
        Initialize the anomaly detection model
        Args:
            df: DataFrame containing the data
            feature_names: List of feature column names
            target_names: List of target column names
            stock: Stock symbol
            verbose: Control logging verbosity
        """
        super().__init__(df, feature_names, target_names, stock)
        self.verbose = verbose
        self._setup_logging()
        
        self.models: Dict[str, ModelType] = {}
        self.model_training: Dict[str, NDArray[np.float64]] = {}
        self.training_preds: Dict[str, pd.DataFrame] = {}
        self.test_preds: Dict[str, pd.DataFrame] = {}
        self.decomp: Dict[str, NDArray[np.float64]] = {}
        self.decomp_preds: Dict[str, NDArray[np.float64]] = {}
        self.centers: Dict[str, NDArray[np.int64]] = {}
        self.distances: Dict[str, NDArray[np.float64]] = {}
        
        logger.debug("Initializing model instance")
        self.initialize(*kwargs)
    
    def merge_preds(self, trainpred, testpred, model_name='anomaly'):
        """Merge predictions with price data"""
        if not isinstance(trainpred, (pd.Series, pd.DataFrame)):
            trainpred = pd.DataFrame(trainpred, index=self.ytrain.index, columns=[model_name])
            testpred = pd.DataFrame(testpred, index=self.ytest.index, columns=[model_name])
            
        trainpred = self.price_data.loc[self.xtrain.index].join(trainpred)
        testpred = self.price_data.loc[self.xtest.index].join(testpred)
        return trainpred, testpred
    
    def pca_pred(self, pred):
        """Handle dimensionality reduction predictions"""
        if not isinstance(pred, (pd.Series, pd.DataFrame)):
            pred = pd.DataFrame(pred, index=self.features_scaled.index)
            
        trainpred = self.price_data.loc[self.xtrain.index].join(pred.loc[self.xtrain.index])
        testpred = self.price_data.loc[self.xtest.index].join(pred.loc[self.xtest.index])
        return trainpred, testpred
    
    
    def _setup_logging(self) -> None:
        """Configure logging based on verbosity"""
        log_level = DEBUG if self.verbose else INFO
        logger.setLevel(log_level)
        
    def _log_value_counts(self, predictions: ArrayLike, model_name: str) -> None:
        """Log value counts from model predictions"""
        if isinstance(predictions, (pd.Series, pd.DataFrame)):
            counts = predictions.value_counts()
        else:
            counts = pd.Series(predictions).value_counts()
        normal = counts.get(1, 0)
        anomalies = counts.get(-1, 0)
        logger.debug(f"{model_name} predictions - Normal: {normal}, Anomalies: {anomalies}")
        
    def _isolation_forest(self) -> None:
        """Unsupervised Isolation Forest for anomaly detection"""
        logger.debug("Running Isolation Forest...")
            
        iso = IsolationForest(contamination='auto', random_state=999)
        iso.fit(self.xtrain)
        
        # Predict (-1 for outliers, 1 for inliers)
        train_pred = iso.predict(self.xtrain)
        test_pred = iso.predict(self.xtest)
        
        self.training_preds['IsolationForest'], self.test_preds['IsolationForest'] = \
            self.merge_preds(train_pred, test_pred, model_name="IsolationForest")
            
        self._log_value_counts(test_pred, "IsolationForest")
        
        self.models['IsolationForest'] = iso
        self.model_training['IsolationForest'] = iso.score_samples(self.xtrain)
    
    def _optimize_svm_params(self) -> Dict[str, Any]:
        """
        Optimize SVM parameters using grid search with custom anomaly scorer
        Returns:
            Dictionary of best parameters
        """
        logger.debug("Optimizing SVM parameters...")
        
        # Define parameter grid
        param_grid = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'degree': [2, 3, 4]  # For polynomial kernel
        }
        
        # Initialize base model
        base_model = OneClassSVM()
        
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Custom scorer for anomaly detection
        def anomaly_scorer(estimator, X, y=None):
            """
            Custom scoring function for anomaly detection
            Args:
                estimator: Fitted estimator
                X: Input data
                y: True labels (optional, included for compatibility)
            Returns:
                float: Combined metric score
            """
            try:
                # Get decision function scores (negative for anomalies)
                scores = estimator.decision_function(X)
                
                # Convert to binary predictions (-1 for anomaly, 1 for normal)
                # Using median as threshold (common approach for OneClassSVM)
                threshold = np.median(scores)
                y_pred = np.where(scores <= threshold, -1, 1)
                
                # If true labels are provided, calculate supervised metrics
                if y is not None and len(y) == len(y_pred):
                    precision = precision_score(y, y_pred, pos_label=-1, zero_division=0)
                    recall = recall_score(y, y_pred, pos_label=-1, zero_division=0)
                    f1 = f1_score(y, y_pred, pos_label=-1, zero_division=0)
                    # Weighted combination of metrics
                    return 0.5 * f1 + 0.3 * precision + 0.2 * recall
                
                # If no true labels (unsupervised), use silhouette-like metric
                else:
                    # Normalize scores
                    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
                    # Calculate separation between predicted anomalies and normals
                    anomaly_scores = scores_norm[y_pred == -1]
                    normal_scores = scores_norm[y_pred == 1]
                    if len(anomaly_scores) == 0 or len(normal_scores) == 0:
                        return 0.0
                    separation = np.abs(np.mean(anomaly_scores) - np.mean(normal_scores))
                    return separation
                    
            except Exception as e:
                logger.debug(f"Scoring error: {str(e)}")
                return 0.0
        
        # Create scorer
        custom_scorer = make_scorer(
            anomaly_scorer,
            greater_is_better=True,
            needs_threshold=False
        )
        
        # Perform grid search with time series CV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=custom_scorer,
            cv=tscv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        logger.info("Starting SVM grid search...")
        try:
            # Convert to numpy array if DataFrame
            X_train = self.xtrain.values if isinstance(self.xtrain, pd.DataFrame) else self.xtrain
            
            # Check if ytrain exists and is valid
            if hasattr(self, 'ytrain') and len(self.ytrain) == len(self.xtrain):
                grid_search.fit(X_train, self.ytrain)
            else:
                grid_search.fit(X_train)
                
            # Log results
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best score: {grid_search.best_score_:.4f}")
            
            # Log top 3 parameter combinations
            results_df = pd.DataFrame(grid_search.cv_results_)
            top_3 = results_df.nlargest(3, 'mean_test_score')
                
            return grid_search.best_params_
            
        except Exception as e:
            logger.error(f"Grid search failed: {str(e)}")
            return param_grid  # Return default parameters as fallback
        
    def _svm(self) -> None:
        """One Class SVM for anomaly detection"""
        logger.debug("Running One-Class SVM...")
            
        # Optimize parameters
        best_params = self._optimize_svm_params()
        
        # Initialize and fit model with best parameters
        svm = OneClassSVM(**best_params)
        svm.fit(self.xtrain)
        
        # Predict (-1 for outliers, 1 for inliers)
        train_pred = svm.predict(self.xtrain)
        test_pred = svm.predict(self.xtest)
        
        self.training_preds['SVM'], self.test_preds['SVM'] = \
            self.merge_preds(train_pred, test_pred, model_name="SVM")
            
        self._log_value_counts(test_pred, "SVM")
        
        self.models['SVM'] = svm
        self.model_training['SVM'] = svm.score_samples(self.xtrain)
    
    def _lof(self) -> None:
        """Local Outlier Factor for anomaly detection"""
        logger.debug("Running Local Outlier Factor...")
            
        # Create DataFrames with consistent feature names
        feature_cols = self.features_scaled.columns
        
        # Convert training and test data to DataFrames with feature names
        xtrain_df = pd.DataFrame(
            self.xtrain,
            columns=feature_cols,
            index=self.xtrain.index
        )
        xtest_df = pd.DataFrame(
            self.xtest,
            columns=feature_cols,
            index=self.xtest.index
        )
            
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto')
        
        # Fit using DataFrame with feature names
        lof.fit(xtrain_df)
        
        # Predict using DataFrames to maintain feature names
        train_pred = lof.predict(xtrain_df)
        test_pred = lof.predict(xtest_df)
        
        self.training_preds['LOF'], self.test_preds['LOF'] = \
            self.merge_preds(train_pred, test_pred, model_name="LOF")
            
        self._log_value_counts(test_pred, "LOF")
        
        self.models['LOF'] = lof
        # Use DataFrame for score_samples to maintain feature names
        self.model_training['LOF'] = lof.negative_outlier_factor_
    
    def _kmeansAnomalyDetection(
        self, 
        x: ArrayLike,
        threshold: float = 1.1618,
        name: str = 'Kmeans'
    ) -> pd.DataFrame:
        """
        K-Means based anomaly detection
        Args:
            x: Input features
            threshold: Distance threshold for anomaly detection
            name: Model name for storing results
        Returns:
            DataFrame with predictions
        """
        logger.debug(f"Running K-means anomaly detection ({name})...")
            
        # Find optimal number of clusters
        param_grid = {
            'n_clusters': np.arange(2, 6),
            'n_init': [10]  # Explicitly set n_init
        }
        kmeans = KMeans(random_state=42, n_init=10)
        grid_search = GridSearchCV(kmeans, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(x)
        
        # Use best parameters including n_init
        best_params = grid_search.best_params_
        logger.debug(f"Best parameters for {name}: {best_params}")
            
        kmeans_model = KMeans(
            n_clusters=best_params['n_clusters'],
            n_init=best_params['n_init'],
            random_state=42
        )
        kmeans_model.fit(x)
        
        # Store model and get cluster assignments
        self.models[name] = kmeans_model
        centers = kmeans_model.cluster_centers_
        labels = kmeans_model.labels_
        
        # Calculate distances to cluster centers
        distances = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            point = x[i]
            center = centers[labels[i]]
            distances[i] = np.linalg.norm(point - center)
            
        self.distances[name] = distances
        
        # Convert distances to anomaly predictions (-1 for anomaly, 1 for normal)
        predictions = np.where(distances >= threshold, -1, 1)
        self.decomp_preds[name] = predictions
        self.centers[name] = labels
        
        pred_df = pd.DataFrame(predictions, index=self.features_scaled.index, columns=[name])
        self._log_value_counts(pred_df, name)
        return pred_df
    
    def _find_optimal_threshold(
        self, 
        distances: NDArray[np.float64],
        name: str
    ) -> float:
        """
        Find optimal threshold for K-means anomaly detection
        Args:
            distances: Array of distances from points to cluster centers
            name: Model name for logging
        Returns:
            Optimal threshold value
        """
        thresholds = np.linspace(0, 2, 100)
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = (distances >= threshold).astype(int)
            score = f1_score(self.ytrain, predictions)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
                
        logger.debug(f"{name} - Best threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
        return best_threshold
    
    def _pca(self) -> PCA:
        """
        PCA dimensionality reduction
        Returns:
            Fitted PCA model
        """
        logger.debug("Running PCA...")
        # Force 2 components
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(self.features_scaled)
        self.decomp['PCA'] = transformed
        
        variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        logger.debug(f"PCA component variance ratios: {variance_ratio}")
        logger.debug(f"Cumulative explained variance: {cumulative_variance}")
        
        return pca
    

    def _kmeans_pca(self, threshold: float = 1.1618) -> None:
        """K-means anomaly detection in PCA space"""
        pca = self._pca()
        transformed = self.decomp['PCA']
        
        # Ensure we have 2D data for visualization
        if transformed.shape[1] < 2:
            logger.warning("PCA produced less than 2 components, padding with zeros")
            pad_width = ((0, 0), (0, 2 - transformed.shape[1]))
            transformed = np.pad(transformed, pad_width, mode='constant')
            
        predictions = self._kmeansAnomalyDetection(transformed, threshold=threshold, name='PCA')
        
        # Store first two components with predictions
        components = pd.DataFrame(
            transformed[:, :2], 
            columns=['PC1', 'PC2'], 
            index=predictions.index
        )
        results = components.join(predictions)
        
        self.training_preds['PCA'], self.test_preds['PCA'] = self.pca_pred(predictions)
        
    
    def _ica_detection(self, threshold: float = 1.1618) -> None:
        """Independent Component Analysis with K-means anomaly detection"""
        logger.debug("Running ICA detection...")
            
        ica = FastICA(n_components=2, random_state=999)
        transformed = ica.fit_transform(self.features_scaled)
        self.decomp['ICA'] = transformed
        
        predictions = self._kmeansAnomalyDetection(transformed, threshold=threshold, name='ICA')
        
        components = pd.DataFrame(
            transformed,
            columns=['IC1', 'IC2'],
            index=predictions.index
        )
        results = components.join(predictions)
        
        self.training_preds['ICA'], self.test_preds['ICA'] = self.pca_pred(predictions)

    
    def fit(self, threshold: float = 1.1618) -> None:
        """
        Fit all anomaly detection models
        Args:
            threshold: Distance threshold for K-means based methods
        """
        if not hasattr(self, 'features_scaled'):
            raise ValueError("Data not initialized. Call initialize() first.")
            
        # Traditional anomaly detection methods
        self._isolation_forest()
        self._svm()
        self._lof()
        
        # Dimensionality reduction + K-means methods
        self._kmeans_pca(threshold)
        self._ica_detection(threshold)
        
        self.decomp_models = ['PCA','ICA']
        
        logger.info("Model fitting complete")
        for name in self.models:
            logger.debug(
                f"{name} predictions shape - "
                f"Train: {self.training_preds[name].shape}, "
                f"Test: {self.test_preds[name].shape}"
            )
