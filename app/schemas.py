from pydantic import BaseModel, Field
from typing import List

class PredictResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    gradcam_url: str

class PredictBatchItem(BaseModel):
    filename: str
    label: str
    confidence: float
    gradcam_url: str

class ModelsResponseItem(BaseModel):
    id: str
    name: str
    params: str
    input_size: str
    trained_on: str
    roc_auc: float

class Health(BaseModel):
    status: str
