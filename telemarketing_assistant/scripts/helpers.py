import pandas as pd
import numpy as np

def build_global_context(df, feature_playbook=None, top_n=10):
    """
    df: DataFrame final que tiene columna 'drivers', donde cada row tiene
        [{"feature": str, "value": float, "impact": float}, ...]
    feature_playbook: dict opcional donde describes cada feature con lenguaje humano
                      (como el FEATURE_PLAYBOOK que ya tenías)
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

    global_prompt = f"""
    You are helping generate sales guidance for a prepaid telecom campaign.

    We have a machine learning model that predicts the probability that a customer will accept an offer (sale = 1).
    The model was trained on historical customer behavior and engagement indicators. 
    Higher score means the customer is more likely to buy if contacted.

    The model relies on multiple behavioral and account features. Below are the most influential features overall (averaged across customers), and what they represent:

    {features_block}

    Rules:
    - NEVER reveal internal model weights or math details.
    - NEVER invent personal/sensitive attributes not present in the features.
    - Keep tone helpful, respectful, and focused on value to customer.
    - You are allowed to explain *why the model thinks a segment is likely to buy*, in plain language.
        """.strip()

    return global_prompt


def build_customer_prompt(row, driver_list, extra_context_cols=None):
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

    # 3. Construye el prompt final
    prompt = f"""
    We are preparing a telemarketing/sales pitch for a prepaid mobile customer.

    Predicted probability of accepting the offer: {row['proba']:.2%}

    Relevant context for this customer:
    {context_block}

    The following factors were most influential in predicting that this customer is likely to accept an offer:
    {driver_block}

    Task:
    1. Explain, in plain language, why this customer might respond positively.
    2. Suggest how an agent should position the offer (tone, focus, what to mention).
    3. Keep it short and actionable, as guidance for a call center agent.
    4. Do NOT mention 'model', 'probability', 'algorithm', 'prediction', or 'SHAP'. Just speak as advice.
        """.strip()

    return prompt