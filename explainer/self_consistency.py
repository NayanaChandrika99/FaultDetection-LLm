"""
Self-Consistency Explanation Generation
Generates multiple explanations and votes for the most consistent diagnosis.
"""

import json
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
from .llm_setup import LLMExplainer
from .prompt_templates import (
    SYSTEM_PROMPT,
    create_explanation_prompt,
    validate_explanation_format
)


def parse_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract and parse JSON from LLM response.
    Handles DeepSeek-style responses with reasoning text.
    
    Args:
        text: Raw LLM output text
    
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Clean up LaTeX math notation that breaks JSON
    text = text.replace('\\(', '').replace('\\)', '')
    text = text.replace('\\[', '').replace('\\]', '')
    
    # Try direct parsing first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strip whitespace
    text = text.strip()
    
    # Try to find JSON object by looking for first { and last }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace:last_brace+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON in markdown code blocks
    if '```json' in text or '```' in text:
        # Extract content between ```json and ``` or ``` and ```
        patterns = [
            (text.find('```json'), text.find('```', text.find('```json') + 7)),
            (text.find('```'), text.find('```', text.find('```') + 3))
        ]
        
        for start_marker_idx, end_marker_idx in patterns:
            if start_marker_idx != -1 and end_marker_idx != -1:
                # Find the actual JSON content
                content_start = text.find('{', start_marker_idx)
                content_end = text.rfind('}', content_start, end_marker_idx) + 1
                
                if content_start != -1 and content_end > content_start:
                    json_str = text[content_start:content_end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
    
    return None


def generate_single_explanation(
    llm_explainer: LLMExplainer,
    features: Dict[str, float],
    prediction: str,
    confidence: float,
    temperature: float = 0.8
) -> Optional[Dict]:
    """
    Generate a single explanation attempt.
    
    Args:
        llm_explainer: LLM explainer instance
        features: Extracted features
        prediction: Primary classifier prediction
        confidence: Prediction confidence
        temperature: Sampling temperature
    
    Returns:
        Parsed explanation dict or None if failed
    """
    # Create prompt
    user_prompt = create_explanation_prompt(features, prediction, confidence)
    
    # Generate
    try:
        response = llm_explainer.explain(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature
        )
        
        # Parse JSON
        explanation = parse_json_from_text(response)
        
        if explanation is None:
            print(f"  ✗ JSON parsing failed. Response preview: {response[:200]}...")
            return None
        
        # Validate format
        is_valid, error_msg = validate_explanation_format(explanation)
        
        if not is_valid:
            print(f"  ✗ Validation failed: {error_msg}")
            return None
        
        return explanation
        
    except Exception as e:
        print(f"  ✗ Generation error: {str(e)}")
        return None


def explain_with_self_consistency(
    llm_explainer: LLMExplainer,
    features: Dict[str, float],
    prediction: str,
    confidence: float,
    k: int = 5,
    temperature: float = 0.8,
    max_attempts: int = 10
) -> Dict:
    """
    Generate explanation using self-consistency (vote over k samples).
    
    Args:
        llm_explainer: LLM explainer instance
        features: Extracted features
        prediction: Primary classifier prediction
        confidence: Prediction confidence
        k: Number of explanations to generate
        temperature: Sampling temperature
        max_attempts: Max attempts to get k valid explanations
    
    Returns:
        Consolidated explanation with voting results
    
    Process:
        1. Generate k explanations with temperature > 0
        2. Vote on final_diagnosis (most common)
        3. Aggregate evidence from agreeing explanations
        4. Return consolidated result with confidence
    """
    print(f"\nGenerating {k} explanations with self-consistency...")
    
    explanations = []
    attempts = 0
    
    while len(explanations) < k and attempts < max_attempts:
        attempts += 1
        print(f"  Attempt {attempts}...", end=" ")
        
        explanation = generate_single_explanation(
            llm_explainer=llm_explainer,
            features=features,
            prediction=prediction,
            confidence=confidence,
            temperature=temperature
        )
        
        if explanation is not None:
            explanations.append(explanation)
            print(f"✓ Valid ({len(explanations)}/{k})")
        else:
            print("✗ Invalid (see error above)")
    
    if len(explanations) == 0:
        print("  WARNING: No valid explanations generated, returning fallback")
        return {
            'final_diagnosis': prediction,
            'confidence': confidence,
            'evidence': [
                f"Primary classifier predicted {prediction} with {confidence:.2%} confidence",
                "No valid LLM explanation could be generated",
                "Using classifier prediction as fallback"
            ],
            'cross_checks': [],
            'recommended_actions': ["Review sensor data quality", "Consider manual inspection"],
            'meta': {
                'n_valid_explanations': 0,
                'n_attempts': attempts,
                'voting_agreement': 0.0,
            }
        }
    
    # Vote on final diagnosis
    diagnoses = [exp['final_diagnosis'] for exp in explanations]
    diagnosis_counts = Counter(diagnoses)
    final_diagnosis, vote_count = diagnosis_counts.most_common(1)[0]
    voting_agreement = vote_count / len(explanations)
    
    print(f"\nVoting results:")
    for diag, count in diagnosis_counts.most_common():
        print(f"  {diag}: {count}/{len(explanations)} ({count/len(explanations)*100:.1f}%)")
    
    # Get explanations that agree with the voted diagnosis
    agreeing_explanations = [
        exp for exp in explanations 
        if exp['final_diagnosis'] == final_diagnosis
    ]
    
    # Aggregate confidence from agreeing explanations
    avg_confidence = np.mean([exp['confidence'] for exp in agreeing_explanations])
    
    # Aggregate evidence (unique claims from agreeing explanations)
    all_evidence = []
    for exp in agreeing_explanations:
        all_evidence.extend(exp['evidence'])
    
    # Deduplicate evidence (keep most frequent, up to 5)
    evidence_counts = Counter(all_evidence)
    top_evidence = [evidence for evidence, _ in evidence_counts.most_common(5)]
    
    # Aggregate cross-checks
    all_cross_checks = []
    for exp in agreeing_explanations:
        all_cross_checks.extend(exp.get('cross_checks', []))
    cross_check_counts = Counter(all_cross_checks)
    top_cross_checks = [check for check, _ in cross_check_counts.most_common(3)]
    
    # Aggregate actions
    all_actions = []
    for exp in agreeing_explanations:
        all_actions.extend(exp.get('recommended_actions', []))
    action_counts = Counter(all_actions)
    top_actions = [action for action, _ in action_counts.most_common(3)]
    
    # Build consolidated explanation
    consolidated = {
        'final_diagnosis': final_diagnosis,
        'confidence': float(avg_confidence),
        'evidence': top_evidence,
        'cross_checks': top_cross_checks,
        'recommended_actions': top_actions,
        'meta': {
            'n_valid_explanations': len(explanations),
            'n_attempts': attempts,
            'voting_agreement': float(voting_agreement),
            'primary_classifier': prediction,
            'primary_confidence': float(confidence),
            'all_diagnoses': dict(diagnosis_counts),
        }
    }
    
    print(f"\nFinal diagnosis: {final_diagnosis} (confidence: {avg_confidence:.3f})")
    print(f"Agreement: {voting_agreement:.1%}")
    
    return consolidated


def batch_explain_with_self_consistency(
    llm_explainer: LLMExplainer,
    features_list: List[Dict[str, float]],
    predictions: List[str],
    confidences: List[float],
    k: int = 5,
    temperature: float = 0.8
) -> List[Dict]:
    """
    Generate explanations for multiple windows in batch.
    
    Args:
        llm_explainer: LLM explainer instance
        features_list: List of feature dicts
        predictions: List of predictions
        confidences: List of confidences
        k: Number of explanations per window
        temperature: Sampling temperature
    
    Returns:
        List of consolidated explanations
    """
    explanations = []
    
    print(f"\nGenerating explanations for {len(features_list)} windows...")
    
    for i, (features, pred, conf) in enumerate(zip(features_list, predictions, confidences)):
        print(f"\n{'='*60}")
        print(f"Window {i+1}/{len(features_list)}")
        print(f"{'='*60}")
        
        explanation = explain_with_self_consistency(
            llm_explainer=llm_explainer,
            features=features,
            prediction=pred,
            confidence=conf,
            k=k,
            temperature=temperature
        )
        
        explanations.append(explanation)
    
    return explanations

