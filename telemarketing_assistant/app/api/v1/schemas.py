from pydantic import BaseModel
from typing import List, Dict, Any

class CustomerRequest(BaseModel):
    customer_id: str
    objective: str  # e.g. "reactivation", "upsell_data", "loyalty"
    channel: str    # e.g. "sms", "call", "push"

class FeatureDriver(BaseModel):
    feature: str
    value: Any
    impact: float

class CustomerResponse(BaseModel):
    customer_id: str
    generated_message: str
    top_features: List[FeatureDriver]