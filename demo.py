#!/usr/bin/env python3
"""
FD-LLM Interactive Demo
=======================

Demonstrates the complete fault detection and explanation system.

Usage:
    python demo.py
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import sys

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")


def print_section(text):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}▶ {text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*80}{Colors.ENDC}")


def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_info(text, indent=0):
    """Print info message"""
    prefix = "  " * indent
    print(f"{prefix}{text}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def load_model_info():
    """Load and display model information"""
    print_section("1. Loading Trained Model")
    
    model_path = Path("outputs/exp_full_dataset/model.pkl")
    
    if not model_path.exists():
        print_error(f"Model not found at {model_path}")
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print_success(f"Model loaded from {model_path}")
    print_info(f"Model type: {type(model).__name__}", 1)
    
    return model


def load_predictions():
    """Load and analyze predictions"""
    print_section("2. Analyzing Classifier Predictions")
    
    pred_path = Path("outputs/exp_full_dataset/predictions_for_colab.parquet")
    
    if not pred_path.exists():
        print_error(f"Predictions not found at {pred_path}")
        return None
    
    df = pd.read_parquet(pred_path)
    
    print_success(f"Loaded {len(df)} predictions")
    print_info(f"Columns: {', '.join(df.columns[:5])}...", 1)
    
    # Analyze prediction distribution
    pred_counts = df['prediction'].value_counts()
    
    print_info("\nPrediction Distribution:", 1)
    for pred, count in pred_counts.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print_info(f"{pred:25s}: {count:5d} ({pct:5.1f}%) {bar}", 2)
    
    # Confidence statistics
    print_info(f"\nConfidence Statistics:", 1)
    print_info(f"Mean: {df['confidence'].mean():.3f}", 2)
    print_info(f"Median: {df['confidence'].median():.3f}", 2)
    print_info(f"Min: {df['confidence'].min():.3f}", 2)
    print_info(f"Max: {df['confidence'].max():.3f}", 2)
    
    return df


def load_explanations():
    """Load and analyze LLM explanations"""
    print_section("3. Loading LLM Explanations")
    
    exp_path = Path("outputs/exp_full_dataset/explanations.jsonl")
    
    if not exp_path.exists():
        print_error(f"Explanations not found at {exp_path}")
        return None
    
    explanations = []
    with open(exp_path, 'r') as f:
        for line in f:
            explanations.append(json.loads(line))
    
    print_success(f"Loaded {len(explanations)} explanations")
    
    # Analyze explanation quality
    diagnoses = [exp['final_diagnosis'] for exp in explanations]
    agreements = [exp['meta']['voting_agreement'] for exp in explanations]
    
    print_info(f"\nExplanation Quality:", 1)
    print_info(f"Average voting agreement: {np.mean(agreements):.1%}", 2)
    print_info(f"Perfect agreement (100%): {sum(1 for a in agreements if a == 1.0)} / {len(agreements)}", 2)
    
    # Top diagnoses
    diag_counts = Counter(diagnoses)
    print_info(f"\nTop Diagnoses from LLM:", 1)
    for diag, count in diag_counts.most_common(5):
        pct = (count / len(explanations)) * 100
        print_info(f"{diag:25s}: {count:5d} ({pct:5.1f}%)", 2)
    
    return explanations


def show_example_explanation(explanations, predictions_df):
    """Show a detailed example explanation"""
    print_section("4. Example Fault Explanation")
    
    # Find a high-confidence fault with good explanation
    for exp in explanations:
        if exp['confidence'] > 0.8 and exp['meta']['voting_agreement'] == 1.0:
            if exp['final_diagnosis'] not in ['Normal']:
                example = exp
                break
    else:
        # Fallback to first explanation
        example = explanations[0]
    
    window_id = example['window_id']
    
    # Get classifier prediction for this window
    pred_row = predictions_df[predictions_df.index == window_id].iloc[0]
    
    print_info(f"Window ID: {window_id}")
    print_info(f"Timestamp: {example.get('timestamp', 'N/A')}")
    print()
    
    print_info(f"{Colors.BOLD}Classifier Prediction:{Colors.ENDC}")
    print_info(f"  Class: {Colors.OKGREEN}{pred_row['prediction']}{Colors.ENDC}", 1)
    print_info(f"  Confidence: {pred_row['confidence']:.1%}", 1)
    print()
    
    print_info(f"{Colors.BOLD}LLM Explanation:{Colors.ENDC}")
    print_info(f"  Diagnosis: {Colors.OKGREEN}{example['final_diagnosis']}{Colors.ENDC}", 1)
    print_info(f"  Confidence: {example['confidence']:.1%}", 1)
    print_info(f"  Agreement: {example['meta']['voting_agreement']:.0%} ({example['meta']['n_valid_explanations']}/5 votes)", 1)
    print()
    
    print_info(f"{Colors.BOLD}Evidence:{Colors.ENDC}")
    for i, evidence in enumerate(example['evidence'][:5], 1):
        print_info(f"  {i}. {evidence}", 1)
    
    if len(example['evidence']) > 5:
        print_info(f"  ... and {len(example['evidence']) - 5} more", 1)
    print()
    
    print_info(f"{Colors.BOLD}Cross-Checks:{Colors.ENDC}")
    for check in example['cross_checks'][:3]:
        print_info(f"  • {check}", 1)
    print()
    
    print_info(f"{Colors.BOLD}Recommended Actions:{Colors.ENDC}")
    for action in example['recommended_actions'][:3]:
        print_info(f"  → {action}", 1)
    print()


def show_classifier_llm_agreement(predictions_df, explanations):
    """Analyze agreement between classifier and LLM"""
    print_section("5. Classifier vs LLM Agreement Analysis")
    
    # Create mapping
    exp_dict = {exp['window_id']: exp for exp in explanations}
    
    agreements = []
    disagreements = []
    
    for idx, row in predictions_df.iterrows():
        if idx in exp_dict:
            classifier_pred = row['prediction']
            llm_pred = exp_dict[idx]['final_diagnosis']
            
            if classifier_pred == llm_pred:
                agreements.append((idx, classifier_pred))
            else:
                disagreements.append((idx, classifier_pred, llm_pred))
    
    total = len(agreements) + len(disagreements)
    agreement_rate = len(agreements) / total if total > 0 else 0
    
    print_info(f"Total windows analyzed: {total}")
    print_info(f"Agreement: {len(agreements)} ({agreement_rate:.1%})")
    print_info(f"Disagreement: {len(disagreements)} ({(1-agreement_rate):.1%})")
    print()
    
    if disagreements:
        print_info("Example Disagreements (first 5):", 1)
        for window_id, clf_pred, llm_pred in disagreements[:5]:
            print_info(f"Window {window_id}: Classifier={clf_pred}, LLM={llm_pred}", 2)


def show_system_capabilities():
    """Display system capabilities"""
    print_section("6. System Capabilities")
    
    capabilities = [
        ("Time-Series Classification", "MultiROCKET + Ridge Classifier", "✓"),
        ("Feature Engineering", "Flow, Density, Pressure features", "✓"),
        ("Physical Validation", "Mass balance, Density-SG checks", "✓"),
        ("LLM Explanations", "Mistral-7B-Instruct with QLoRA", "✓"),
        ("Self-Consistency", "5-vote ensemble for reliability", "✓"),
        ("Fault Classes", "10 classes (Normal + 9 faults)", "✓"),
        ("Real-time Capable", "Classifier <1ms, LLM ~3s per window", "✓"),
        ("Colab Integration", "GPU-accelerated explanation generation", "✓"),
    ]
    
    for capability, description, status in capabilities:
        print_info(f"{status} {Colors.BOLD}{capability:30s}{Colors.ENDC} - {description}")


def show_performance_metrics():
    """Display model performance metrics"""
    print_section("7. Model Performance")
    
    results_path = Path("outputs/exp_full_dataset/results.json")
    
    if not results_path.exists():
        print_warning("Results file not found, skipping metrics")
        return
    
    # Read just the first part of the file to get summary metrics
    with open(results_path, 'r') as f:
        content = f.read(5000)  # Read first 5KB
        try:
            results = json.loads(content)
        except:
            print_warning("Could not parse results file")
            return
    
    print_info("Classifier Performance:")
    if 'test_accuracy' in results:
        print_info(f"  Accuracy: {results['test_accuracy']:.1%}", 1)
    if 'test_f1_macro' in results:
        print_info(f"  Macro F1: {results['test_f1_macro']:.3f}", 1)
    
    print_info("\nNote: This is a demo project with 75% accuracy", 1)
    print_info("Production deployment would require 90%+ accuracy", 1)


def show_next_steps():
    """Show next steps for users"""
    print_section("8. Next Steps")
    
    print_info("To use this system:")
    print()
    print_info("1. Train on your data:", 1)
    print_info("   python training/train_rocket.py --data your_data.csv --run_name exp_001", 2)
    print()
    print_info("2. Export for Colab:", 1)
    print_info("   python scripts/export_for_colab.py --model outputs/exp_001/model.pkl --data your_data.csv", 2)
    print()
    print_info("3. Generate explanations on Colab GPU:", 1)
    print_info("   - Upload FD_LLM_Colab_Explainer.ipynb to Colab", 2)
    print_info("   - Select GPU runtime and run all cells", 2)
    print()
    print_info("4. Analyze results:", 1)
    print_info("   python scripts/analyze_classifier_performance.py", 2)
    print()
    print_info("For more details, see:", 1)
    print_info("  • README.md - Main documentation", 2)
    print_info("  • DEMO_PROJECT_SUMMARY.md - Project status", 2)
    print_info("  • docs/COLAB_SETUP.md - Colab integration guide", 2)


def main():
    """Main demo function"""
    print_header("FD-LLM: Fault Detection with LLM Explanations - DEMO")
    
    print(f"{Colors.OKCYAN}This demo showcases the complete hybrid system:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  • Fast time-series classifier for real-time detection{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  • LLM explainer for natural language explanations{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  • Self-consistency voting for reliability{Colors.ENDC}")
    
    try:
        # Load model
        model = load_model_info()
        if model is None:
            print_error("\nCannot continue without model. Please train a model first.")
            print_info("Run: python training/train_rocket.py --data data/raw/your_data.csv")
            return 1
        
        # Load predictions
        predictions_df = load_predictions()
        if predictions_df is None:
            print_error("\nCannot continue without predictions.")
            return 1
        
        # Load explanations
        explanations = load_explanations()
        if explanations is None:
            print_warning("\nNo explanations found. Continuing with classifier results only.")
            explanations = []
        
        # Show example explanation
        if explanations:
            show_example_explanation(explanations, predictions_df)
            show_classifier_llm_agreement(predictions_df, explanations)
        
        # Show capabilities
        show_system_capabilities()
        
        # Show performance
        show_performance_metrics()
        
        # Show next steps
        show_next_steps()
        
        print_header("Demo Complete!")
        print(f"\n{Colors.OKGREEN}✓ Successfully demonstrated FD-LLM system{Colors.ENDC}")
        print(f"{Colors.OKCYAN}For questions or issues, see the documentation in docs/{Colors.ENDC}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Demo interrupted by user{Colors.ENDC}")
        return 130
    except Exception as e:
        print_error(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

