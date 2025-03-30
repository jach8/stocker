"""
Anomaly detection model using various algorithms
"""
from __future__ import annotations
import os
import logging
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats 
from logging import getLogger, Logger, DEBUG, INFO, Formatter, StreamHandler
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
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline

import sys
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from connect import setup

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning,message='X does not have valid feature names')

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

    def save_models(self, directory: str = "bin/price/models/anom/saved") -> None:
        """Save all fitted models to disk"""
        if not self.models:
            logger.error("No models to save. Fit the model first.")
            return
        if not self.distances: 
            logger.error("No distances to save. Fit the model first.")
            return
        
        os.makedirs(directory, exist_ok=True)
        if hasattr(self, 'stock') and self.stock:  # Per-stock saving
            filepath = os.path.join(directory, f"anomaly_model_{self.stock}.pkl")
            joblib.dump({
                'scaler': self.scaler,
                'models': self.models,
                'decomp': self.decomp, 
                'distances': self.distances,
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'stock': self.stock
            }, filepath)
            logger.info(f"Saved models for {self.stock} to {filepath}")
        else:  # Single model for all
            filepath = os.path.join(directory, "anomaly_model_all_stocks.pkl")
            joblib.dump({
                'scaler': self.scaler,
                'models': self.models,
                'decomp': self.decomp,
                'distances': self.distances,
                'feature_names': self.feature_names,
                'target_names': self.target_names
            }, filepath)
            logger.info(f"Saved combined model to {filepath}")

    def load_and_predict(self, new_data: pd.DataFrame, stock: str = None, directory: str = "bin/price/models/anom/saved/") -> Dict[str, pd.DataFrame]:
        """
        Load saved models and predict on new data
        Args:
            new_data: New stock data with same features
            stock: Stock symbol (if per-stock models)
            directory: Where models are saved
        Returns:
            Dictionary of model names to prediction DataFrames
        """
        # Determine which model to load
        if stock and os.path.exists(os.path.join(directory, f"anomaly_model_{stock}.pkl")):
            filepath = os.path.join(directory, f"anomaly_model_{stock}.pkl")
        else:
            filepath = os.path.join(directory, "anomaly_model_all_stocks.pkl")
            stock = None
        
        # Load model
        try:
            saved_data = joblib.load(filepath)
            self.models = saved_data['models']
            self.decomp = saved_data['decomp']
            self.scaler = saved_data['scaler']
            self.distances = saved_data['distances']
            self.feature_names = saved_data['feature_names']
            self.target_names = saved_data['target_names']
            if stock:
                self.stock = saved_data['stock']
            logger.info(f"Loaded models from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
        
        # Prepare new data
        new_features = new_data[self.feature_names]
        if hasattr(self, 'scaler'):  # Assuming you have a scaler from setup
            new_features_scaled = self.scaler.transform(new_features)
        else:
            new_features_scaled = new_features.values
        
        # Predict with each model
        predictions = {}
        for model_name, model in self.models.items():
            try:
                if model_name in ['PCA', 'ICA']:  # Handle decomposition models
                    transformed = self._transform_decomp(new_features_scaled, model_name)
                    distances = self._compute_kmeans_distances(transformed, model_name)
                    pred = np.where(distances >= self.distances[model_name].mean(), -1, 1)
                else:
                    pred = model.predict(new_features_scaled)
                predictions[model_name] = pd.DataFrame(pred, index=new_data.index, columns=[model_name])
                self._log_value_counts(pred, model_name)
            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {str(e)}")
        
        return predictions

    def _transform_decomp(self, data, model_name):
        """Transform new data using stored decomposition"""
        assert model_name in self.decomp, f"Decomposition model {model_name} not found"
        return self.decomp[model_name].transform(data)

    def _compute_kmeans_distances(self, transformed, model_name):
        """Compute distances for K-Means based models"""
        try:
            centers = self.models[model_name].cluster_centers_
            labels = self.models[model_name].predict(transformed)
            distances = np.array([np.linalg.norm(transformed[i] - centers[labels[i]]) 
                                for i in range(len(transformed))])
            return distances
        except Exception as e:
            logger.error(f"Failed to compute distances for {model_name}: {str(e)}")
            raise
    
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
            
        iso = IsolationForest(contamination=0.05, random_state=999)
        iso.fit(self.xtrain)
        
        # Predict (-1 for outliers, 1 for inliers)
        train_pred = iso.predict(self.xtrain)
        test_pred = iso.predict(self.xtest)
        
        self.training_preds['IsolationForest'], self.test_preds['IsolationForest'] = \
            self.merge_preds(train_pred, test_pred, model_name="IsolationForest")
            
        self._log_value_counts(test_pred, "IsolationForest")
        
        self.models['IsolationForest'] = iso
        self.model_training['IsolationForest'] = iso.score_samples(self.xtrain)
    
    def _svm(self) -> None:
        """One Class SVM for anomaly detection"""
        logger.debug("Running One-Class SVM...")
            
        # Optimize parameters
        def optimize_svm_params() -> Dict[str, Any]:
            """
            Optimize SVM parameters using grid search with custom anomaly scorer
            Returns:
                Dictionary of best parameters
            """
            logger.debug("Optimizing First SVM parameters...")
            
            # Define parameter grid
            param_grid = {
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3, 4],  # For polynomial kernel
                # 'average': [None, 'micro', 'macro', 'weighted']
            }
            
            # Initialize base model
            base_model = OneClassSVM()
            
            # Set up time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Define custom scorer for anomaly detection
            scorer = make_scorer(f1_score, average = 'micro')

            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring=scorer,
                n_jobs=-1
            )
            grid_search.fit(self.xtrain, self.ytrain)

            # Get best parameters
            best_params = grid_search.best_params_

            return best_params
        
        best_params = optimize_svm_params()
        
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
            

        def optimize_lof_params(xtrain_df, ytrain_df):
            """
            Optimize LOF parameters using grid search with custom anomaly scorer
            Returns:
                Dictionary of best parameters
            """
            logger.debug("Optimizing LOF parameters...")
            
            # Define parameter grid
            param_grid = {
                'n_neighbors': np.arange(6, 28, 2),
                'contamination': [0.05,0.06,0.07,0.08,0.09,0.1] ,
                'novelty': [True]
            }
            
            # Initialize base model
            base_model = LocalOutlierFactor()
            
            # Set up time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Define custom scorer for anomaly detection
            scorer = make_scorer(f1_score, average = 'macro')

            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring=scorer,
                n_jobs=-1
            )
            grid_search.fit(xtrain_df, ytrain_df)

            # Get best parameters
            return grid_search.best_params_
        
        # Create DataFrames with consistent feature names
        feature_cols = self.features_scaled.columns
        
        best_params = optimize_lof_params(self.xtrain.values, self.ytrain.values)
        lof = LocalOutlierFactor(**best_params)
        
        # Fit using DataFrame with feature names
        lof.fit(self.xtrain.values)
        
        # Predict using DataFrames to maintain feature names
        train_pred = lof.predict(self.xtrain.values)
        test_pred = lof.predict(self.xtest.values)
        
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
        avg_dist = distances.mean()
        threshold = avg_dist * 1.5
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
        pca = PCA(n_components=2).fit(self.features_scaled)
        self.decomp['PCA'] = pca
        variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        logger.debug(f"PCA component variance ratios: {variance_ratio}")
        logger.debug(f"Cumulative explained variance: {cumulative_variance}")
        
        return pca
    
    def _kmeans_pca(self, threshold: float = 1.1618) -> None:
        """K-means anomaly detection in PCA space"""
        pca = self._pca()
        transformed = self.decomp['PCA'].transform(self.features_scaled)
        
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
            
        ica = FastICA(n_components=2, random_state=999).fit(self.features_scaled)
        self.decomp['ICA'] = ica
        transformed = ica.transform(self.features_scaled)
        
        predictions = self._kmeansAnomalyDetection(transformed, threshold=threshold, name='ICA')
        
        components = pd.DataFrame(
            transformed,
            columns=['IC1', 'IC2'],
            index=predictions.index
        )
        results = components.join(predictions)
        
        self.training_preds['ICA'], self.test_preds['ICA'] = self.pca_pred(predictions)

    def _svm_sgd(self, nu: float = 0.05, gamma: float = 2.0, n_components: int = 100) -> None:
        """
        One-Class SVM with kernel approximation using Nystroem and SGD optimization
        Args:
            nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors
            gamma: Kernel coefficient for 'rbf' kernel
            n_components: Number of components to keep in Nystroem approximation
        """
        logger.debug("Running One-Class SVM with kernel approximation and SGD...")
        
        # Set random state for reproducibility
        random_state = 999
        
        # Create kernel approximation
        transform = Nystroem(
            kernel='rbf',
            gamma=gamma,
            n_components=n_components,
            random_state=random_state
        )
        
        # Create SGD-based One-Class SVM
        sgd_ocsvm = SGDOneClassSVM(
            nu=nu,
            shuffle=True,
            fit_intercept=True,
            random_state=random_state,
            tol=1e-4,
            learning_rate='optimal'
        )
        
        # Create pipeline
        model = make_pipeline(transform, sgd_ocsvm)
        
        # Fit the model
        try:
            # Convert to numpy array if DataFrame
            X_train = self.xtrain.values if isinstance(self.xtrain, pd.DataFrame) else self.xtrain
            
            model.fit(X_train)
            
            # Predict
            train_pred = model.predict(self.xtrain)
            test_pred = model.predict(self.xtest)
            
            # Store predictions
            self.training_preds['SVM_SGD'], self.test_preds['SVM_SGD'] = \
                self.merge_preds(train_pred, test_pred, model_name="SVM_SGD")
            
            # Log prediction statistics
            self._log_value_counts(test_pred, "SVM_SGD")
            
            # Store model and scores
            self.models['SVM_SGD'] = model
            self.model_training['SVM_SGD'] = model.decision_function(self.xtrain)
            
            # Log additional information
            n_errors_train = np.sum(train_pred == -1)
            n_errors_test = np.sum(test_pred == -1)
            logger.info(f"SVM_SGD Training errors: {n_errors_train}")
            logger.info(f"SVM_SGD Test errors: {n_errors_test}")
            
        except Exception as e:
            logger.error(f"Error fitting SVM_SGD: {str(e)}")
            raise
        
        logger.debug("SVM_SGD fitting completed successfully")
    
    def _svm_gd_tuned(self) -> None:
        """One-Class SVM with grid search for hyperparameter tuning"""
        logger.debug("Running One-Class SVM with grid search...")
        
        # Define parameter grid
        param_grid = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.2],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4],  # For polynomial kernel
        }
        
        # Initialize base model
        base_model = OneClassSVM()
        
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define custom scorer for anomaly detection we only want about 5% of the data to be anomalies
        scorer = make_scorer(f1_score, average = 'micro')


        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring=scorer,
            n_jobs=-1
        )
        grid_search.fit(self.xtrain, self.ytrain)

        # Get best parameters
        best_params = grid_search.best_params_
        logger.debug(f"Best parameters for SVM_GD: {best_params}")
        
        # Initialize and fit model with best parameters
        svm = OneClassSVM(**best_params)
        svm.fit(self.xtrain)
        
        # Predict (-1 for outliers, 1 for inliers)
        train_pred = svm.predict(self.xtrain)
        test_pred = svm.predict(self.xtest)
        
        self.training_preds['SVM_GD'], self.test_preds['SVM_GD'] = \
            self.merge_preds(train_pred, test_pred, model_name="SVM_GD")
            
        self._log_value_counts(test_pred, "SVM_GD")
        
        self.models['SVM_GD'] = svm
        self.model_training['SVM_GD'] = svm.score_samples(self.xtrain)

    def fit(self, threshold: float = 1.1618, **kwargs) -> None:
        """
        Fit all anomaly detection models
        Args:
            threshold: Distance threshold for K-means based methods
        """
        logger.debug("Initializing model instance")
        self.initialize(*kwargs)
            
        # Traditional anomaly detection methods
        self._isolation_forest()
        self._svm()
        self._lof()
        self._svm_gd_tuned()
        
        # Dimensionality reduction + K-means methods
        self._kmeans_pca(threshold)
        self._ica_detection(threshold)
        
        self.decomp_models = ['PCA','ICA']
        
        logger.info("Model fitting complete")
        for name in self.models:
            try:
                training_preds = self.training_preds[name]
                test_preds = self.test_preds[name]
                self._log_value_counts(training_preds, name)
                self._log_value_counts(test_preds, name)
            except Exception as e:
                logger.error(f"Failed to log value counts for {name}: {str(e)}")
