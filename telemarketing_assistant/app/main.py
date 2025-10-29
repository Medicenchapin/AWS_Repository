from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import router as api_router


# Create principal FastAPI instance 
app = FastAPI(
    title="Telemarketing LLM Assistant API",
    description="Backend API for customer explanation and personalized message generation using SHAP + LLM",
    version="1.0.0",
)


# CORS Configuration
origins = [
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Principal endpoints registred
app.include_router(api_router, prefix="/api/v1")


# Basic test endpoint
@app.get("/")
def read_root():
    return {
        "message": "🚀 Telemarketing LLM Assistant is running",
        "docs": "/docs",
        "api_version": "v1",
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}


# 5. Entry endpoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
