import os
import sys
import json
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.app.ml.classifier import FakeNewsClassifierPipeline
from backend.app.core.config import settings

def main():
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    test_path = os.path.join(BASE_DIR, "data", "processed", "test.csv")
    models_dir = os.path.join(BASE_DIR, "backend", "models")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Data files not found. Running preprocessor first...")
        from scripts.preprocess_data import main as run_preprocess
        run_preprocess()
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()
    
    model_types = ["naive_bayes", "logistic_regression", "svm", "random_forest"]
    results = {}
    best_f1 = -1.0
    best_pipeline = None
    
    print(f"--- Training Classical Baselines on {len(X_train)} samples ---")
    
    for m_type in model_types:
        print(f"\nTraining {m_type}...")
        pipeline = FakeNewsClassifierPipeline(model_type=m_type)
        pipeline.fit(X_train, y_train)
        
        metrics = pipeline.evaluate(X_test, y_test)
        results[m_type] = metrics
        
        print(f"Results for {m_type}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Save every model to models directory
        pipeline.save(models_dir)
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_pipeline = pipeline

    # Save evaluation summary to JSON
    eval_json_path = os.path.join(BASE_DIR, "data", "processed", "model_evaluation.json")
    with open(eval_json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nModel evaluation metrics saved to: {eval_json_path}")
    print(f"Best model: {best_pipeline.model_type} with F1 = {best_f1:.4f}")
    
    # Save best as primary active model
    best_pipeline.save(models_dir)
    print(f"Saved primary production model to {models_dir}")

if __name__ == "__main__":
    main()
