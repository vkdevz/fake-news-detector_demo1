import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

def main():
    eval_json_path = os.path.join(BASE_DIR, "data", "processed", "model_evaluation.json")
    if not os.path.exists(eval_json_path):
        print("Running train_baseline.py first...")
        from scripts.train_baseline import main as run_train
        run_train()
        
    with open(eval_json_path, "r") as f:
        metrics = json.load(f)
        
    print("\n" + "="*70)
    print("      TRUTHLENS: EMPIRICAL MACHINE LEARNING MODEL COMPARISON")
    print("="*70)
    
    header = f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}"
    print(header)
    print("-" * len(header))
    
    table_rows = []
    for model_name, m in metrics.items():
        name_display = model_name.replace("_", " ").title()
        row = f"{name_display:<22} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['roc_auc']:<10.4f}"
        print(row)
        table_rows.append({
            "model": name_display,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "confusion_matrix": m["confusion_matrix"]
        })
        
    print("="*70 + "\n")
    return table_rows

if __name__ == "__main__":
    main()
