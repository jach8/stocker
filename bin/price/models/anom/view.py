import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from logging import getLogger

import sys
from .model import anomaly_model as models

# Configure module logger
logger = getLogger(__name__)

class viewer(models):
    """Visualization class for anomaly detection results"""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        feature_names: List[str], 
        target_names: List[str], 
        stock: str, 
        verbose: bool = False
    ) -> None:
        """
        Initialize the anomaly visualization class
        Args:
            df: DataFrame containing the data
            feature_names: List of feature column names
            target_names: List of target column names
            stock: Stock symbol
            verbose: Control logging verbosity
        """
        super().__init__(df, feature_names, target_names, stock, verbose)
        logger.debug(f"Initializing viewer for stock {stock}")
        
    def general_plot(self) -> None:
        """Generate visualization plots for all anomaly detection methods"""
        logger.info("Generating general visualization plots")
        
        all_preds = list(self.training_preds.keys())
        n = len(all_preds)
        logger.debug(f"Creating plots for {n} prediction methods")
        
        fig, ax = plt.subplots(2, 3, figsize=(20, 6))
        ax = ax.flatten()
        ii = 0 

        # Plotting test predictions for each method
        for i, p in enumerate(all_preds[:]):
            logger.debug(f"Plotting results for {p}")
            cs = self.test_preds[p]['close']
            sc = self.test_preds[p]
            ax[ii].plot(self.test_preds[p]['close'], label='close')
            ax[ii].scatter(
                self.test_preds[p].index, 
                self.test_preds[p]['close'], 
                c=self.test_preds[p][p]
            )
            ax[ii].set_title(f'{p} Anomalies')
            ax[ii].legend()
            ii += 1
                    
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        
        # Process final predictions
        all_preds = list(self.training_preds.keys())
        logger.debug("Processing final predictions")
        lodf: List[pd.Series] = []
        for i in all_preds:
            # Append the last test prediction
            lodf.append(self.test_preds[i][i])

        self.last_pred = pd.concat(lodf, axis=1, keys=all_preds).tail(1)
        logger.debug(f"Final prediction shape: {self.last_pred.shape}")