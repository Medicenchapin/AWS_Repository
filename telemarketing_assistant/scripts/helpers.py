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


def build_customer_prompt(row, driver_list, extra_context_cols=None, name_field=None):
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
    Using ONLY the context above and the campaign rules from system prompt:
    1) Decide Eligibility: {{Yes/No}} and give a brief reason if "No".
    2) Select Suggested Band/Plan: {{PLAN_Q115|PLAN_Q135|PLAN_Q160|PLAN_Q185|PLAN_Q209}}.
    3) Provide Authorized offer range: {{Qxx.xx–Qyy.yy}}.
    4) Recommend an initial price within the authorized range: {{Qxx.xx}}. Justify in one line referencing recent spend.
    5) If applicable, propose an Upsell option (next band) with a one-line justification.

    Then produce a concise 3-line agent script:
    - Value: tie benefits to recent spend (“with what you already invest per month…”).
    - Price: keep within the assigned range.
    - Close: immediate activation, no contract, same line/top-ups.

    [Output Format]
    Eligibility: <Yes/No> (+ reason if No)
    Suggested Plan: <PLAN_Q115|PLAN_Q135|PLAN_Q160|PLAN_Q185|PLAN_Q209>
    Authorized range: <Qxx.xx–Qyy.yy>
    Recommended price: <Qxx.xx>  # one-line justification
    Upsell option: <plan_if_any>  # brief justification
    Script:
    “{name_value}, over the last 3 months you’ve invested about Q<arpu_3m_prom>/month.
    With the <plan_sugerido> plan you get more data/minutes for Q<precio_recomendado>, keeping your usual spend but with more value.
    Shall I confirm activation? It goes live today — no contract, and you keep your same line.”
    """.strip()

    return prompt