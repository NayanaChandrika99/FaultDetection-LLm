"""
LLM Prompt Templates
Structured prompts for fault diagnosis explanation.
"""

from typing import Dict, Optional, Tuple


SYSTEM_PROMPT = """You are an expert process engineer specializing in slurry pipeline systems. 
Your role is to analyze sensor data and explain fault diagnoses clearly and accurately.

CRITICAL INSTRUCTION: You must respond with ONLY valid JSON. No explanations, no reasoning text, no markdown formatting. Just pure JSON starting with { and ending with }.

You will be provided with:
1. A primary classifier's prediction and confidence
2. Key sensor measurements and derived features
3. Physical constraints and thresholds

Your task is to:
- Analyze the diagnosis internally
- Provide evidence grounded in the actual sensor values
- Output ONLY valid JSON in the specified format (no other text)

Requirements:
- Include at least 3 specific numeric claims in your evidence
- Reference actual sensor values and thresholds
- Provide actionable recommendations
- Be concise and precise
- RESPOND WITH ONLY JSON - NO OTHER TEXT"""


def create_explanation_prompt(
    features: Dict[str, float],
    prediction: str,
    confidence: float,
    sensor_thresholds: Optional[Dict] = None
) -> str:
    """
    Create a structured prompt for LLM explanation.
    
    Args:
        features: Dictionary of extracted features
        prediction: Primary classifier's prediction
        confidence: Prediction confidence (0-1)
        sensor_thresholds: Optional dict of threshold values
    
    Returns:
        Formatted prompt string
    
    Expected output format:
        {
            "final_diagnosis": "class_name",
            "confidence": 0.0-1.0,
            "evidence": ["claim1", "claim2", "claim3", ...],
            "cross_checks": ["check1", "check2", ...],
            "recommended_actions": ["action1", "action2", ...]
        }
    """
    if sensor_thresholds is None:
        sensor_thresholds = {
            'cv_unstable': 0.15,
            'zero_flow': 0.01,
            'sg_deviation': 0.03,
            'density_spike_sigma': 3.0,
            'pressure_variability': 0.1,
            'dv_drift': 5.0,
        }
    
    # Helper function to format numeric values
    def fmt(key, default=0.0, decimals=3):
        val = features.get(key, default)
        if val is None or (isinstance(val, str) and val == 'N/A'):
            return 'N/A'
        try:
            return f"{float(val):.{decimals}f}"
        except (ValueError, TypeError):
            return 'N/A'
    
    # Format sensor data section
    flow_section = f"""FLOW MEASUREMENTS:
  Mean Flow: {fmt('flow_mean', decimals=3)} m³/s
  Flow Std Dev: {fmt('flow_std', decimals=3)} m³/s
  Coefficient of Variation: {fmt('flow_cv', decimals=3)} (threshold: {sensor_thresholds['cv_unstable']})
  Zero-Flow Events: {features.get('flow_n_zero_events', 0)}
  Rate of Change: {fmt('flow_rate_of_change', decimals=4)} m³/s per min
  Flow Stability: {fmt('flow_stability', decimals=3)}"""
    
    density_section = f"""DENSITY & SOLIDS:
  Density Mean: {fmt('density_mean', decimals=1)} kg/m³
  Density Std Dev: {fmt('density_std', decimals=1)} kg/m³
  Density Trend: {fmt('density_trend', decimals=2)} kg/m³ per 5min
  Density Spikes (>3σ): {features.get('density_spike_count', 0)}
  SG Mean: {fmt('sg_mean', decimals=3)}
  SG Target Deviation: {fmt('sg_target_deviation', decimals=3)} (threshold: ±{sensor_thresholds['sg_deviation']})
  Percent Solids: {fmt('percent_solids_mean', decimals=1)}%"""
    
    process_section = f"""PROCESS VARIABLES:
  Pressure Mean: {fmt('pressure_mean', decimals=1)} kPa
  Pressure Variability: {fmt('pressure_variability', decimals=3)} (threshold: {sensor_thresholds['pressure_variability']})
  Pressure Spikes: {features.get('pressure_spike_count', 0)}
  Pressure-Flow Correlation: {fmt('pressure_flow_correlation', decimals=3)}
  Temperature: {fmt('temp_mean', decimals=1)}°C
  DV (Particle Size): {fmt('dv_mean', decimals=1)} μm
  DV Drift: {fmt('dv_drift', decimals=1)} μm (segregation threshold: {sensor_thresholds['dv_drift']} μm)"""
    
    mass_balance_section = ""
    if 'mass_flow_mean' in features:
        flow_val = features.get('flow_mean', 0)
        density_val = features.get('density_mean', 0)
        try:
            calculated = float(flow_val) * float(density_val)
            mass_balance_section = f"""MASS BALANCE:
  Measured Mass Flow: {fmt('mass_flow_mean', decimals=1)} kg/s
  Calculated (Flow × Density): {calculated:.1f} kg/s"""
        except (ValueError, TypeError):
            mass_balance_section = f"""MASS BALANCE:
  Measured Mass Flow: {fmt('mass_flow_mean', decimals=1)} kg/s
  Calculated (Flow × Density): N/A kg/s"""
    
    prompt = f"""PRIMARY CLASSIFIER PREDICTION:
  Diagnosis: {prediction}
  Confidence: {confidence:.3f}

{flow_section}

{density_section}

{process_section}

{mass_balance_section}

TASK:
Analyze the sensor data and provide a structured explanation for the fault diagnosis.
Consider the primary classifier's prediction but use your engineering judgment.

CRITICAL REQUIREMENTS:
1. MUST respond with ONLY valid JSON (start with {{, end with }})
2. Evidence MUST contain at least 3 claims with ACTUAL NUMBERS from the data above
3. Each evidence claim MUST reference specific sensor values with comparison operators (>, <, =, ±)
4. Use PLAIN TEXT only - NO LaTeX, NO markdown, NO math notation like \( or \)

EXAMPLE EVIDENCE (use actual values from data above):
- "Density mean at 1009.6 kg/m³ is below normal range (expected >1050 kg/m³)"
- "Density trending downward at -65 kg/m³ per 5min indicates progressive dilution"
- "SG at 1.010 is 0.020 below target (threshold: ±0.03), confirming water addition"

OUTPUT FORMAT (use ACTUAL numbers from sensor data):
{{
    "final_diagnosis": "Dilution",
    "confidence": 0.85,
    "evidence": [
        "Density mean at X kg/m³ compared to Y threshold",
        "Flow CV at X (threshold: 0.15) indicates stability/instability",
        "SG deviation of X from target (threshold: ±0.03)",
        "Density trend at X kg/m³ per 5min shows increasing/decreasing pattern"
    ],
    "cross_checks": [
        "Mass balance check: measured vs calculated flow",
        "Density-SG correlation verification"
    ],
    "recommended_actions": [
        "Specific operational action based on diagnosis",
        "Monitoring recommendation for key parameter"
    ]
}}

Valid diagnosis options: Normal, Dilution, Settling/Segregation, Concentration, Air Entrainment, Pipeline Blockage, Pump Cavitation, Valve Transient, Instrument Fault, Process Upset"""
    
    return prompt


def validate_explanation_format(explanation: Dict) -> Tuple[bool, str]:
    """
    Validate that explanation follows required format.
    
    Args:
        explanation: Dict from LLM output
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # SIMPLIFIED VALIDATION:
    # 1. Check if the structure is correct (all keys present).
    # 2. Check that at least 2 evidence claims contain ANY number.
    
    # 1. Structure check
    required_keys = ['final_diagnosis', 'confidence', 'evidence', 'cross_checks', 'recommended_actions']
    for key in required_keys:
        if key not in explanation:
            return False, f"Missing required key: '{key}'"
            
    if not isinstance(explanation.get('evidence'), list) or len(explanation['evidence']) < 3:
        return False, f"Evidence must be a list with at least 3 claims. Got: {explanation.get('evidence')}"

    # 2. Numeric evidence check (lenient)
    numeric_claims = 0
    for claim in explanation.get('evidence', []):
        if any(char.isdigit() for char in str(claim)):
            numeric_claims += 1
            
    if numeric_claims < 2:
        return False, f"Evidence requires at least 2 numeric claims. Found {numeric_claims}. Evidence: {explanation.get('evidence')}"

    return True, "valid"


# Fault class descriptions for reference
FAULT_DESCRIPTIONS = {
    'Normal': 'Normal operating conditions with stable flow, density, and pressure',
    'Pump Cavitation': 'Low inlet pressure causing vapor bubbles, flow fluctuations',
    'Pipeline Blockage': 'Increased pressure, decreased flow, possible density increase',
    'Settling/Segregation': 'Particle size drift, density stratification, uneven solids distribution',
    'Air Entrainment': 'Decreased density and SG, flow oscillations, pressure instability',
    'Dilution': 'Decreased density and percent solids, SG below target',
    'Concentration': 'Increased density and percent solids, SG above target',
    'Valve Transient': 'Sudden flow changes, zero-flow events, pressure spikes',
    'Instrument Fault': 'Physical inconsistencies, mass balance violations, implausible readings',
    'Process Upset': 'Multiple parameters out of range, unstable conditions',
}

