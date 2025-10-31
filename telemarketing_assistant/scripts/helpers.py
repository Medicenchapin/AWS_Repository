import pandas as pd
import numpy as np

def build_global_context(df, feature_playbook=None, top_n=10, rules_text=None):
    """
    Builds a concise, production-ready SYSTEM prompt.
    - df: DataFrame with a 'drivers' column (list[dict]: {'feature','value','impact'})
    - feature_playbook: dict[feature] -> human description (optional)
    - top_n: how many globally most-influential features to list
    - rules_text: optional custom rules block (str). If None, defaults to ARPU band logic.
    """
    # Expandimos todos los drivers en un solo dataframe
    drivers_all = (
        df["drivers"]
        .explode()
        .dropna()
        .apply(pd.Series)  # -> columns: feature, value, impact
    )

    # Importancia global: mean(|impact|)
    global_importance = (
        drivers_all
        .groupby("feature")["impact"]
        .apply(lambda s: s.abs().mean())
        .sort_values(ascending=False)
        .head(top_n)
    )

    top_features = list(global_importance.index)

    # Armamos descripción de features
    lines = []
    for feat in top_features:
        if feature_playbook and feat in feature_playbook:
            desc = feature_playbook[feat]
        else:
            desc = "No description available."
        lines.append(f"- {feat}: {desc}")

    features_block = "\n".join(lines)

    # Default business rules (editable)
    if rules_text is None:
        rules_text = """
        Business Rules (apply consistently):
        1) Window: last 3 full months (M-1, M-2, M-3).
        2) Monthly ARPU = net revenue paid by the customer (top-ups, bundles, add-ons). Exclude freebies/bonuses, chargebacks, and adjustments.
        3) Eligibility: consumption > 0 in each of the 3 months, ARPU_3M_PROM ≥ Q80.00, and no commercial blocks.
        4) Offer mapping by ARPU_3M_PROM:
        • Q80.00–Q110.99 → PLAN_Q115
        • Q111.00–Q130.99 → PLAN_Q135
        • Q131.00–Q155.99 → PLAN_Q160
        • Q156.00–Q180.99 → PLAN_Q185
        • ≥ Q181.00       → PLAN_Q209
        5) Controlled upsell: if ARPU_3M_PROM is in the top 10% of its band and all three monthly ARPUs are ≥ 90% of the next band’s lower bound, offer the next band as an alternative.
        6) Downsell: on price objection, offer the minimum of the current band’s range (do not cross down a band unless affordability constraints are explicit).
        7) Messaging: emphasize benefits, keep price within the assigned band, and anchor value to actual spending.
        """.strip()

    global_prompt = f"""
    You are an expert sales advisor for a prepaid telecom campaign. Maximize conversions with sustainable offers aligned to each customer's real consumption.

    We use a machine learning model trained on historical behavior and engagement to estimate the probability of accepting an offer (sale=1). A higher score means higher likelihood if contacted.

    Most influential features overall (by mean absolute impact):
    {features_block}

    {rules_text}

    Policy:
        - Do NOT reveal internal model weights or math.
    - Do NOT invent personal/sensitive attributes beyond provided data.
    - Keep tone helpful, respectful, and value-focused.
     - You may explain drivers in plain language, but never mention “model”, “probability”, or “SHAP” in the final agent script.
    """.strip()

    return global_prompt


def _build_feature_points(driver_list, feature_playbook=None):
    """
    Construye los 5 puntos principales basados en los drivers y el playbook de características
    
    Args:
        driver_list: Lista de diccionarios con información de drivers
        feature_playbook: Diccionario de características y sus descripciones
    
    Returns:
        str: Los 5 puntos formateados como texto
    """
    # Ordenar drivers por impacto absoluto
    sorted_drivers = sorted(driver_list, key=lambda x: abs(x['impact']), reverse=True)
    
    points = []
    for i, driver in enumerate(sorted_drivers[:5], 1):
        feature = driver['feature']
        description = feature_playbook.get(feature, "No description available.") if feature_playbook else ""
        points.append(f"{i}) {feature}: {description}")
    
    return "\n".join(points)

def build_customer_prompt(row, driver_list, extra_context_cols=None, name_field=None, feature_playbook=None):
    """
    row: una fila de tu df (por ejemplo df.loc[idx])
         que tiene columnas humanas tipo 'state_name', 'previous_classification', etc.
         y también 'proba'.

    driver_list: lista de dicts tipo:
      [
        {"feature": "arpu_90_days", "value": 8.24, "impact": 0.32},
        ...
      ]

    extra_context_cols: lista de columnas del row que quieres pasar al LLM
                        para que hable más personalizado.
                        Ej: ["state_name", "previous_classification", "arpu_90_days", "network_age_years"]
    
    feature_playbook: diccionario con descripciones de las características
    """

    # 1. Prepara resumen de los drivers SHAP personalizados de este cliente
    driver_lines = []
    for d in driver_list:
        driver_lines.append(
            f"- {d['feature']}: value={d['value']}, impact={d['impact']:+.3f}"
        )
    driver_block = "\n".join(driver_lines)

    # 2. Agrega contexto humano del cliente (opcional)
    context_lines = []
    if extra_context_cols:
        for col in extra_context_cols:
            if col in row:
                context_lines.append(f"{col} = {row[col]}")
    context_block = "\n".join(context_lines) if context_lines else "No additional context."
    
    # Optional name for sample script
    name_value = (row.get(name_field) if name_field and name_field in row else "Customer")

    # 3. Construye el prompt final
    prompt = f"""
    [Customer Context]
    - Acceptance likelihood (score): {row.get('proba', float('nan')):.2%}
    - Attributes:
    {context_block}

    - Top influencing factors for this specific customer:
    {driver_block}

    [Task]
    Analyze these top 5 features from highest to lowest impact, using ONLY their values and impacts:

    1) Feature Analysis (Most Impactful):
       - Feature Context: Use the playbook description to understand what this raw metric means
       - Current Value: Interpret the specific value provided
       - Impact Direction: Consider if the impact is positive/negative for conversion
       - Action Insight: Determine specific actions based on this feature's influence

    2) Secondary Pattern:
       - Feature Context: Reference the playbook meaning for this raw metric
       - Value Analysis: Evaluate the concrete measurement provided
       - Impact Interpretation: Understand how this affects customer decision
       - Usage Pattern: Extract behavioral insights from this data point

    3) Supporting Evidence:
       - Feature Context: Apply the playbook definition
       - Data Point Analysis: Break down what the value indicates
       - Impact Assessment: Connect impact direction to customer behavior
       - Pattern Recognition: Identify relevant trends from this metric

    4) Behavioral Marker:
       - Feature Context: Consider the playbook explanation
       - Value Significance: Analyze what this measurement reveals
       - Impact Contribution: How this shapes customer response
       - Behavioral Insight: Extract actionable understanding

    5) Final Indicator:
       - Feature Context: Use the playbook to frame this metric
       - Value Review: What does this specific measurement tell us
       - Impact Role: How this influences overall likelihood
       - Opportunity Signal: Identify potential based on this data

    Then provide a data-driven synthesis:
    - Evidence Summary: Combine insights from ALL 5 features, with their specific values
    - Value Connection: Link each feature's impact to concrete benefits
    - Action Strategy: Propose steps based on the analyzed feature set

    [Output Format]
    Customer Analysis for {name_value}:

    Feature Impact Summary:
    1. Primary Driver: [Feature Name] Value: [Raw Metric] | Key Insight: [Based on Playbook]

    2. Secondary Driver: [Feature Name] |Value: [Raw Metric] | Key Insight: [Based on Playbook]

    3. Tertiary Driver: [Feature Name] | Value: [Raw Metric] | Key Insight: [Based on Playbook]

    4. Fourth Driver: [Feature Name] | Value: [Raw Metric] | Key Insight: [Based on Playbook]

    5. Fifth Driver: [Feature Name] | Value: [Raw Metric] | Key Insight: [Based on Playbook]

    Evidence-Based Summary:
    • Patterns: [Key findings from top 3 features]
    • Profile: [Customer behavior based on values]
    • Actions: [Data-supported recommendations]
    """.strip()

    return prompt