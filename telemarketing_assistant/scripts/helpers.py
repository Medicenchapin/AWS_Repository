import pandas as pd
import numpy as np
import json

class Helpers:
    
    def __init__(self, df, feat_playbook_lang: str = 'ESP', top_k: int = 10):
        self.df = df
        self.feat_playbook_lang = feat_playbook_lang
        self.top_k = top_k
        self.features_playbook = None
    
    
    def get_feat_playbook(self):
        # import sys
        # sys.path.append('../')
        if self.feat_playbook_lang == 'ESP':
            with open("../data/config/feature_playbook_esp.json", "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open("../data/config/feature_playbook_eng.json", "r", encoding="utf-8") as f:
                return json.load(f)


    def top_global_features_from_drivers(
        self,
        group_key: str = "raw_feature",     # use "raw_feature" if available, else falls back to "feature"
        rename_map: dict | None = None,     # e.g., {"arpu_90_days": "arpu_3m_prom"}
    ) -> tuple[pd.DataFrame, str]:
        """
        Compute global importance from df['drivers'] (list[dict]) and build a Markdown block.

        Returns:
        summary_df: columns = ['feature', 'mean_abs_impact'], sorted desc, top_n rows
        features_block: Markdown list with feature name + business desc (if in playbook)
        """

        if "drivers" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'drivers' column.")

        # 1) Explode into a flat DataFrame of drivers
        #    (drivers can be list[dict] or JSON string; normalize)
        def _coerce_list(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except Exception:
                    return None
            return x

        drivers_series = self.df["drivers"].apply(_coerce_list)
        drivers_all = (
            drivers_series.explode().dropna().apply(pd.Series)  # -> feature, impact, raw_feature, ...
        )
        if drivers_all.empty:
            return pd.DataFrame(columns=["feature", "mean_abs_impact"]), "- (No se encontraron drivers)"

        # 2) Choose grouping key (prefer raw_feature for OHE; fallback to feature)
        if group_key not in drivers_all.columns or drivers_all[group_key].isna().all():
            use_key = "feature"
        else:
            use_key = group_key

        # 3) Alias/rename (e.g., arpu_90_days → arpu_3m_prom)
        if rename_map:
            drivers_all[use_key] = drivers_all[use_key].map(lambda v: rename_map.get(v, v))

        # 4) Clean impact to numeric and drop NaNs
        drivers_all["impact"] = pd.to_numeric(drivers_all["impact"], errors="coerce")
        drivers_all = drivers_all.dropna(subset=["impact", use_key])

        if drivers_all.empty:
            return pd.DataFrame(columns=["feature", "mean_abs_impact"]), "- (No se encontraron drivers válidos)"

        # 5) Aggregate: mean(|impact|)
        summary_df = (
            drivers_all.assign(abs_impact=lambda x: x["impact"].abs())
            .groupby(use_key, as_index=False)["abs_impact"].mean()
            .rename(columns={use_key: "feature", "abs_impact": "mean_abs_impact"})
            .sort_values("mean_abs_impact", ascending=False)
            .head(self.top_k)
            .reset_index(drop=True)
        )

        # 6) Build Markdown block using the playbook
        lines = []
        feature_playbook = self.get_feat_playbook()
        for _, r in summary_df.iterrows():
            feat = r["feature"]
            desc = (feature_playbook or {}).get(
                feat, "Sin descripción disponible en el playbook."
            )
            lines.append(f"- {feat}: {desc}")
        features_block = "\n".join(lines) if lines else "- (No se encontraron drivers)"

        return summary_df, features_block


    def build_global_system_prompt_es(
        self,
        top_n: int = 10,
        titulo: str = "Resumen Global de Drivers SHAP",
        reglas_extra: str | None = None,
    ) -> str:
            """
            Construye el prompt GLOBAL (system) en español usando:
            - TOP-N variables globales por mean(|impact|) calculado desde df['drivers']
            - Descripciones del FEATURE_PLAYBOOK
            - Instrucciones para NO mostrar valores numéricos de SHAP, solo dirección/interpretación

            Retorna un string listo para pasar como 'system prompt'.
            """
            summary_df, self.features_playbook = self.top_global_features_from_drivers()
            

            reglas_default = """
            Guía de uso:
            - Estos drivers son los más influyentes a nivel global (calculado con la media del impacto absoluto de SHAP).
            - En resúmenes por cliente, NO muestres valores numéricos de SHAP; solo la dirección (positivo/negativo) y una interpretación breve.
            - Usa lenguaje de negocio, conciso y objetivo; evita especulación o información no presente en los datos.
            - Para variables one-hot, muestra el nombre crudo + categoría (p. ej., previous_classification = NEW_CLIENT).
            - Ordena siempre por relevancia (|impacto|) de mayor a menor.
            """.strip()

            if reglas_extra:
                reglas_block = reglas_default + "\n\nReglas adicionales:\n" + reglas_extra.strip()
            else:
                reglas_block = reglas_default

            system_prompt = f"""
            Eres un asistente analítico para una empresa de telecomunicaciones. Tu función es ayudar a interpretar los principales drivers (valores SHAP) del modelo a nivel global y por cliente, en términos de negocio.

            {titulo}
            Estas son las variables globalmente más influyentes (TOP {top_n}) y su significado de negocio:
            {self.features_playbook}

            Definición breve:
            - SHAP indica la contribución de cada variable a la predicción del modelo para un caso específico.
            - Signo positivo: aumenta la probabilidad del resultado deseado (apoya contacto).
            - Signo negativo: disminuye la probabilidad (sugiere cautela o revisión previa).

            {reglas_block}
            """.strip()

            return system_prompt


    def build_customer_prompt_summary(self, row, driver_list, max_features=10):
        """
        Genera un prompt en español que resume los factores más influyentes
        (drivers SHAP) para un cliente específico.

        Parámetros:
        - row: dict o pandas.Series con información del cliente, incluyendo 'proba'
        - driver_list: lista de dicts con {'feature', 'value', 'impact'}
        - feature_playbook: dict con descripciones de negocio (opcional)
        - max_features: número máximo de drivers a incluir

        Retorna: texto del prompt listo para el modelo LLM
        """

        # 1️⃣ Ordenar drivers por importancia absoluta (impacto)
        top_drivers = sorted(driver_list, key=lambda d: abs(d["impact"]), reverse=True)[:max_features]

        # 2️⃣ Crear resumen de cada driver
        driver_lines = []
        for d in top_drivers:
            feature = d["feature"]
            value = d["value"]
            impact = d["impact"]

            direction = "positivo (favorece contacto)" if impact > 0 else "negativo (revisar antes de contactar)"
            desc = self.get_feat_playbook().get(feature, "Sin descripción disponible.") if self.features_playbook else ""

            driver_lines.append(
                f"- **{feature}** ({direction}): valor = {value:.2f}. {desc}"
            )

        driver_block = "\n".join(driver_lines)

        # 3️⃣ Resumen general del cliente
        prompt = f"""
        Eres un analista de campañas de telecomunicaciones prepago.

        Analiza los factores más influyentes en la probabilidad de compra para un cliente individual, basándote en valores SHAP.
        El modelo predijo una probabilidad de aceptación del **{row['proba']:.1%}**.

        A continuación se listan los principales *drivers* (variables) que explican esta predicción,
        ordenados por relevancia:

        {driver_block}

        Tareas:
        1. Resume brevemente qué podría estar motivando o desmotivando al cliente.
        2. Indica si conviene **contactarlo** o **analizar más información** antes de hacerlo.
        3. No menciones palabras técnicas como “modelo”, “SHAP”, “algoritmo” o “predicción”.
        4. Escribe en lenguaje de negocio claro y objetivo, con tono ejecutivo.
            """.strip()                     

        return prompt
    
    def build_customer_prompt_summary_v2(self, row, driver_list, max_features=10):
        """
        Genera un prompt en español que muestra los principales drivers SHAP
        con sus valores crudos e impacto, para que el modelo genere una opinión ejecutiva.

        Parámetros:
        - row: dict o pandas.Series con información del cliente, incluyendo 'proba'
        - driver_list: lista de dicts con {'feature', 'value', 'crude_value', 'impact'}
        - feature_playbook: dict con descripciones de negocio (opcional)
        - max_features: número máximo de drivers a incluir

        Retorna: texto del prompt listo para enviar al LLM
        """

        # 1️⃣ Ordenar drivers por importancia absoluta
        top_drivers = sorted(driver_list, key=lambda d: abs(d["impact"]), reverse=True)[:max_features]

        # 2️⃣ Construir listado con valores crudos y contexto
        driver_lines = []
        for d in top_drivers:
            feature = d.get("feature", "")
            value = d.get("value", None)
            crude_value = d.get("crude_value", None)
            impact = d.get("impact", 0.0)

            direction = "positivo (indica mayor probabilidad de respuesta)" if impact > 0 else "negativo (puede reducir la probabilidad)"
            desc = self.get_feat_playbook().get(feature, "Sin descripción disponible.") if self.get_feat_playbook() else ""

            driver_lines.append(
                f"- **{feature}** → valor crudo: `{crude_value}`, transformado: `{value:.3f}`, impacto: {impact:+.3f} → {direction}. {desc}"
            )

        driver_block = "\n".join(driver_lines)

        # 3️⃣ Construcción del prompt
        prompt = f"""
        Eres un analista especializado en comportamiento de clientes de telecomunicaciones prepago.

        A continuación se muestra un resumen de las variables más influyentes en la predicción de aceptación de oferta
        para un cliente específico. Cada variable incluye su valor crudo, valor transformado y dirección del impacto
        según el modelo analizado.

        Probabilidad estimada de aceptación: **{row['proba']:.1%}**

        **Principales factores del cliente:**
        {driver_block}

        Con base en esta información, proporciona una breve interpretación ejecutiva sobre:
        - Qué comportamiento general refleja este cliente.
        - Qué aspectos podrían estar impulsando o limitando su disposición a aceptar una oferta.

        Tu respuesta debe ser objetiva y realista, evitando lenguaje técnico o especulativo.
        """.strip()

        return prompt
    
    
    def build_customer_prompt_with_shap(
        self,
        row,
        driver_list,
        max_features=10,
        include_json_mirror=True
    ):
        """
        Construye un prompt (en español) para que el LLM devuelva:
        - Viñetas con los TOP-K drivers (con SHAP, valor crudo y transformado)
        - (Opcional) Un bloque JSON espejo con los mismos campos

        Parámetros
        ----------
        row : dict o pandas.Series
            Debe incluir 'proba' (probabilidad estimada) y, opcionalmente, columnas de contexto.
        driver_list : list[dict]
            Cada dict debe tener al menos: 'feature', 'impact'.
            Idealmente incluye: 'raw_feature', 'ohe_category', 'raw_value', 'value' (transformado).
        feature_playbook : dict|None
            Diccionario opcional de descripciones de negocio por variable.
        max_features : int
            Número máximo de drivers a listar.
        include_json_mirror : bool
            Si True, solicita al LLM incluir también un bloque JSON espejo.

        Retorna
        -------
        str : prompt listo para enviar al LLM.
        """

        # 1) Ordena por importancia absoluta
        top_drivers = sorted(driver_list, key=lambda d: abs(d.get("impact", 0.0)), reverse=True)[:max_features]

        # 2) Construye payload limpio para el LLM (redondea números para lectura)
        def _round(x, n=4):
            try:
                return round(float(x), n)
            except Exception:
                return x

        drivers_payload = []
        for d in top_drivers:
            feat = d.get("feature")
            raw_feat = d.get("raw_feature", None)
            ohe_cat = d.get("ohe_category", None)
            raw_val = d.get("raw_value", None)
            val = d.get("value", None)              # transformado
            imp = d.get("impact", None)             # SHAP

            display = None
            if ohe_cat and raw_feat:
                display = f"{raw_feat} = {ohe_cat}"
            elif raw_feat:
                display = raw_feat
            else:
                display = feat

            desc = self.get_feat_playbook().get(display, self.get_feat_playbook().get(feat, "")) if self.get_feat_playbook() else ""

            drivers_payload.append({
                "feature_display": display,
                "feature": feat,
                "raw_feature": raw_feat,
                "ohe_category": ohe_cat,
                "raw_value": raw_val,
                "transformed_value": _round(val, 6) if isinstance(val, (int, float)) else val,
                "shap_value": _round(imp, 6) if isinstance(imp, (int, float)) else imp,
                "direction": "positivo" if (isinstance(imp, (int, float)) and imp > 0) else "negativo",
                "business_hint": desc
            })

        drivers_json = json.dumps(drivers_payload, ensure_ascii=False)

        # 3) Arma el prompt
        proba_txt = f"{float(row['proba'])*100:.1f}%" if "proba" in row else "N/D"
        json_clause = (
            f"""
    Además del listado en viñetas, devuelve a continuación un bloque JSON que sea un ESPEJO EXACTO
    de los {len(top_drivers)} elementos (mismos campos y valores), con la clave raíz "drivers". No agregues otros campos.
            """.strip()
            if include_json_mirror else ""
        )

        prompt = f"""
    Eres un asistente analítico para campañas de telecomunicaciones prepago.

    Cliente:
    - Probabilidad estimada de aceptación: {proba_txt}

    A continuación tienes los **10 principales drivers SHAP** del cliente (ya ordenados por relevancia).
    Cada elemento incluye: nombre para mostrar, valor crudo, valor transformado, valor SHAP y una pista de negocio.
    **No inventes ni alteres valores numéricos**: utiliza exactamente los provistos.

    DRIVERS_JSON:
    {drivers_json}

    Instrucciones de salida (en español):
    1) Devuelve primero un título: "### Resumen SHAP del Cliente".
    2) Luego, una lista de viñetas (una por driver) con este formato:
    - <feature_display>: <dirección (positivo/negativo)>. SHAP=<shap_value>. Crudo=<raw_value>. Transformado=<transformed_value>. Insight=<una frase corta y realista, basada en "business_hint" si está disponible>.
    * Usa exactamente los valores provistos para SHAP, Crudo y Transformado (no los redondees de nuevo).
    * La "dirección" es "positivo" si SHAP > 0, en otro caso "negativo".
    * La frase de insight debe ser breve, de negocio y no técnica. No uses jerga del modelo.
    {json_clause}

    Políticas:
    - No muestres información personal no provista.
    - No inventes métricas ni valores.
    - Mantén el tono profesional y conciso.
    """.strip()

        return prompt
