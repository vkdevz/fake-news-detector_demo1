import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from backend.app.ml.preprocessor import clean_news_text

class FakeNewsClassifierPipeline:
    def __init__(self, model_type: str = "logistic_regression", max_features: int = 5000):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            stop_words='english'
        )
        self.model = self._init_model(model_type)
        self.is_fitted = False

    def _init_model(self, model_type: str):
        if model_type == "logistic_regression":
            return LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        elif model_type == "naive_bayes":
            return MultinomialNB(alpha=0.1)
        elif model_type == "svm":
            # Calibrated LinearSVC provides well-calibrated posterior probabilities
            base_svm = LinearSVC(C=1.0, random_state=42)
            return CalibratedClassifierCV(estimator=base_svm, cv=3)
        elif model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X_train: list, y_train: list):
        cleaned_X = [clean_news_text(text) for text in X_train]
        X_vec = self.vectorizer.fit_transform(cleaned_X)
        self.model.fit(X_vec, y_train)
        self.is_fitted = True

    def evaluate(self, X_test: list, y_test: list) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        
        cleaned_X = [clean_news_text(text) for text in X_test]
        X_vec = self.vectorizer.transform(cleaned_X)
        y_pred = self.model.predict(X_vec)
        
        # Probabilities
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_vec)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        else:
            roc_auc = 0.0
            
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        return {
            "model_type": self.model_type,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "test_samples": len(y_test)
        }

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        
        cleaned = clean_news_text(text)
        X_vec = self.vectorizer.transform([cleaned])
        pred_label = int(self.model.predict(X_vec)[0]) # 1 = FAKE, 0 = REAL
        
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_vec)[0]
            fake_prob = float(proba[1])
            real_prob = float(proba[0])
            confidence = float(max(proba))
        else:
            fake_prob = 1.0 if pred_label == 1 else 0.0
            real_prob = 1.0 - fake_prob
            confidence = 0.75
            
        return {
            "label": "FAKE" if pred_label == 1 else "REAL",
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": round(confidence, 4),
            "model_type": self.model_type
        }

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(save_dir, "tfidf_vectorizer.joblib"))
        joblib.dump(self.model, os.path.join(save_dir, f"{self.model_type}_model.joblib"))

    def load(self, save_dir: str):
        vec_path = os.path.join(save_dir, "tfidf_vectorizer.joblib")
        model_path = os.path.join(save_dir, f"{self.model_type}_model.joblib")
        
        if not os.path.exists(vec_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifacts not found in {save_dir}")
            
        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)
        self.is_fitted = True
