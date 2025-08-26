# End-to-End MLOps Sentiment Analysis Pipeline

## Project Overview
This project demonstrates a complete end-to-end sentiment analysis pipeline using a pretrained Hugging Face model, designed for production-style deployment. It includes data loading, model inference, batch predictions, and a minimal FastAPI service to serve predictions via an API.

## Key Features
- Loads and processes the **IMDB movie reviews dataset**.  
- Uses **Hugging Face’s `distilbert-base-uncased-finetuned-sst-2-english`** model for sentiment prediction.  
- Runs batch predictions with confidence scores.  
- Minimal **FastAPI API** included for demonstration of model serving.  
- Fully runnable in **Google Colab**, requiring no local setup.

## Tech Stack & Tools
- Python 3.12  
- Hugging Face Transformers  
- Datasets (Hugging Face)  
- FastAPI & Uvicorn  
- Colab for interactive execution

## Getting Started
1. Clone the repository:  
```bash
git clone https://github.com/<your-username>/mlops-sentiment-pipeline.git
cd mlops-sentiment-pipeline
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Open the notebook (mlops_sentiment_api.ipynb) in Colab or locally to explore and run the pipeline.

5. (Optional) Run FastAPI locally:
```bash
uvicorn app:app --reload
# Access API at http://127.0.0.1:8000/predict
```
