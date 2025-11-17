"""
CLI Tool for Running LLM Explainer
Loads classifier predictions and generates explanations.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from explainer.llm_setup import LLMExplainer
from explainer.self_consistency import explain_with_self_consistency, batch_explain_with_self_consistency
from models.encoders.feature_extractor import extract_window_features


def load_predictions(pred_file: str):
    """
    Load classifier predictions from file.
    
    Expected format (parquet or CSV):
        - window_id
        - prediction
        - confidence
        - features (JSON string or columns)
    """
    if pred_file.endswith('.parquet'):
        df = pd.read_parquet(pred_file)
    else:
        df = pd.read_csv(pred_file)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate LLM Explanations')
    parser.add_argument('--pred_file', type=str, required=True,
                        help='Path to predictions file (parquet/csv)')
    parser.add_argument('--output', type=str, default='explanations.jsonl',
                        help='Output path for explanations (JSONL format)')
    parser.add_argument('--model_name', type=str, 
                        default='meta-llama/Llama-3-8B-Instruct',
                        help='HuggingFace model identifier')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of explanations for self-consistency')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to process (for testing)')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='Use 4-bit quantization')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (currently only 1 supported)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("FD-LLM EXPLAINER")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Self-consistency k: {args.k}")
    print(f"Temperature: {args.temperature}")
    print(f"4-bit quantization: {args.load_in_4bit}")
    
    # Load predictions
    print(f"\nLoading predictions from: {args.pred_file}")
    pred_df = load_predictions(args.pred_file)
    print(f"Loaded {len(pred_df)} predictions")
    
    # Limit samples if requested
    if args.max_samples:
        print(f"Limiting to {args.max_samples} samples")
        pred_df = pred_df.head(args.max_samples)
    
    # Initialize LLM
    print(f"\nInitializing LLM...")
    llm_explainer = LLMExplainer(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        temperature=args.temperature
    )
    
    # Process each prediction
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    explanations = []
    
    with open(output_path, 'w') as f:
        for idx, row in pred_df.iterrows():
            print(f"\n{'='*60}")
            print(f"Processing {idx+1}/{len(pred_df)}")
            print(f"{'='*60}")
            
            # Extract features
            if 'features' in row and isinstance(row['features'], str):
                # Features stored as JSON string
                features = json.loads(row['features'])
            else:
                # Features stored as columns
                feature_cols = [col for col in pred_df.columns 
                               if col not in ['window_id', 'prediction', 'confidence']]
                features = row[feature_cols].to_dict()
            
            prediction = row['prediction']
            confidence = row['confidence']
            
            print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
            
            # Generate explanation
            explanation = explain_with_self_consistency(
                llm_explainer=llm_explainer,
                features=features,
                prediction=prediction,
                confidence=confidence,
                k=args.k,
                temperature=args.temperature
            )
            
            # Add metadata
            explanation['window_id'] = int(row.get('window_id', idx))
            explanation['timestamp'] = datetime.now().isoformat()
            
            # Write to JSONL
            f.write(json.dumps(explanation) + '\n')
            f.flush()
            
            explanations.append(explanation)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total explanations: {len(explanations)}")
    
    # Count by diagnosis
    from collections import Counter
    diagnosis_counts = Counter([exp['final_diagnosis'] for exp in explanations])
    print("\nDiagnosis distribution:")
    for diag, count in diagnosis_counts.most_common():
        print(f"  {diag}: {count} ({count/len(explanations)*100:.1f}%)")
    
    # Average agreement
    avg_agreement = sum(exp['meta']['voting_agreement'] for exp in explanations) / len(explanations)
    print(f"\nAverage voting agreement: {avg_agreement:.1%}")
    
    # Average confidence
    avg_confidence = sum(exp['confidence'] for exp in explanations) / len(explanations)
    print(f"Average confidence: {avg_confidence:.3f}")
    
    print(f"\nExplanations saved to: {output_path}")
    
    # Free memory
    llm_explainer.free_memory()


if __name__ == '__main__':
    main()

