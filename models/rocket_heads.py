"""
MultiROCKET Classifier
Wrapper for aeon's MultiROCKET with Ridge classifier.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from aeon.transformations.collection.convolution_based import MultiRocket


class MultiROCKETClassifier:
    """
    MultiROCKET time-series classifier with Ridge regression head.
    
    Architecture:
        Input [n_samples, n_channels, length] 
        → MultiROCKET transform [n_samples, n_features]
        → StandardScaler
        → RidgeClassifierCV
        → Predictions
    
    Args:
        n_kernels: Number of ROCKET kernels (default: 6250)
        max_dilations: Maximum dilation for kernels (default: 32)
        alphas: Ridge regularization alphas to try (default: logspace(-3,3,10))
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_kernels: int = 6250,
        max_dilations: int = 32,
        alphas: Optional[np.ndarray] = None,
        random_state: int = 42
    ):
        self.n_kernels = n_kernels
        self.max_dilations = max_dilations
        self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 10)
        self.random_state = random_state
        
        # Initialize components (max_dilations not supported in newer aeon versions)
        self.rocket = MultiRocket(
            n_kernels=n_kernels,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.classifier = RidgeClassifierCV(
            alphas=self.alphas,
            cv=5,
            scoring='accuracy'
        )
        
        self.is_fitted_ = False
        self.classes_ = None
        self.n_features_in_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiROCKETClassifier':
        """
        Fit the MultiROCKET classifier.
        
        Args:
            X: Training data of shape [n_samples, n_channels, length]
            y: Training labels of shape [n_samples]
        
        Returns:
            self
        """
        # Store classes
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Transform with ROCKET
        print(f"Fitting MultiROCKET with {self.n_kernels} kernels...")
        X_transformed = self.rocket.fit_transform(X, y)
        print(f"Transformed shape: {X_transformed.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_transformed)
        
        # Fit classifier
        print(f"Fitting Ridge classifier with {len(self.alphas)} alphas...")
        self.classifier.fit(X_scaled, y)
        print(f"Best alpha: {self.classifier.alpha_}")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted ROCKET.
        
        Args:
            X: Data of shape [n_samples, n_channels, length]
        
        Returns:
            Transformed and scaled features
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transform")
        
        X_transformed = self.rocket.transform(X)
        X_scaled = self.scaler.transform(X_transformed)
        return X_scaled
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for data.
        
        Args:
            X: Data of shape [n_samples, n_channels, length]
        
        Returns:
            Predicted labels of shape [n_samples]
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.transform(X)
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (using decision function).
        
        Args:
            X: Data of shape [n_samples, n_channels, length]
        
        Returns:
            Decision values of shape [n_samples, n_classes]
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.transform(X)
        
        # Ridge classifier doesn't have predict_proba, use decision_function
        decision = self.classifier.decision_function(X_scaled)
        
        # Convert to pseudo-probabilities using softmax
        if len(self.classes_) == 2:
            # Binary case
            decision = np.column_stack([-decision, decision])
        
        # Apply softmax
        exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
        proba = exp_decision / exp_decision.sum(axis=1, keepdims=True)
        
        return proba
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction confidence scores.
        
        Args:
            X: Data of shape [n_samples, n_channels, length]
        
        Returns:
            Confidence scores of shape [n_samples]
        """
        proba = self.predict_proba(X)
        return proba.max(axis=1)
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_dict = {
            'rocket': self.rocket,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'classes_': self.classes_,
            'n_features_in_': self.n_features_in_,
            'is_fitted_': self.is_fitted_,
            'params': {
                'n_kernels': self.n_kernels,
                'max_dilations': self.max_dilations,
                'alphas': self.alphas,
                'random_state': self.random_state,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MultiROCKETClassifier':
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
        
        Returns:
            Loaded MultiROCKETClassifier instance
        """
        with open(path, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Create instance with saved parameters
        params = model_dict['params']
        instance = cls(
            n_kernels=params['n_kernels'],
            max_dilations=params['max_dilations'],
            alphas=params['alphas'],
            random_state=params['random_state']
        )
        
        # Restore fitted components
        instance.rocket = model_dict['rocket']
        instance.scaler = model_dict['scaler']
        instance.classifier = model_dict['classifier']
        instance.classes_ = model_dict['classes_']
        instance.n_features_in_ = model_dict['n_features_in_']
        instance.is_fitted_ = model_dict['is_fitted_']
        
        print(f"Model loaded from {path}")
        return instance
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_kernels': self.n_kernels,
            'max_dilations': self.max_dilations,
            'alphas': self.alphas,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted_,
            'n_classes': len(self.classes_) if self.classes_ is not None else None,
            'n_features_in': self.n_features_in_,
        }

