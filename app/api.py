from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from .mlp_inference import load_model
from .final_inference import predict_complete

# Initialize
app = FastAPI(title="PCB Defect Detection API")

# Import your modules
model, scaler, feature_names = load_model(
    model_path = "./app/multitask_model/best_multitask_model.pth",
    preprocessing_path = "./app/multitask_model/preprocessing_artifacts.pkl",
)

class PredictRequest(BaseModel):
    paste_volume: float
    stencil_thickness: float
    paste_viscosity: float
    ambient_rh: float
    ambient_temperature: float
    
    @field_validator('*')
    @classmethod
    def no_negative_values(cls, v):
        if v < 0:
            raise ValueError("Value cannot be negative")
        return v
    
    class Config:
        extra = 'forbid'  # Reject extra fields

@app.post("/predict")
def predict(request: PredictRequest):
    params = request.dict()
    result = predict_complete(
        params, 
        model, 
        scaler, 
        feature_names, 
        device="cpu"
    )
    return result

@app.get("/health")
def health():
    return {"status": "healthy"}