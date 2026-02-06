from pydantic import BaseModel

class MessageRequest(BaseModel):
    message: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float