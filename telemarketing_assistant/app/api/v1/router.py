from fastapi import APIRouter, HTTPException
from .schemas import CustomerResponse, CustomerRequest

router = APIRouter()

@router.post('explain_customer', response_model=CustomerResponse)
def explain_customer(req: CustomerRequest):
    
    return CustomerResponse(
        customer_id=req.customer_id,
        generated_message="",
        top_features=[]
    )