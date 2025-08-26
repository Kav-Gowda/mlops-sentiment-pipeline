from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Define input schema
class Review(BaseModel):
    text: str

# Load pretrained sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# API endpoint
@app.post("/predict")
def predict_sentiment(review: Review):
    try:
        # Truncate long text for model
        result = sentiment_model(review.text[:512])[0]
        return {"label": result['label'], "score": result['score']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# To run: uvicorn app:app --reload
# Access API at http://127.0.0.1:8000/predict
