from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from utils.io_utils import get_latest_artifact

# Define the application
app = FastAPI(
    title="Naive Bayes Model Server",
    description="A simple API to serve a pretrained Naive Bayes model for text classification.",
    version="0.1.0",
)

# --- Globals ---
# Define artifact paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
# Load the latest model and vectorizer at startup
# We are choosing to serve the MultinomialNB model
MODEL_PATH = get_latest_artifact(MODELS_DIR, "multinomial_nb", exclude_pattern="vectorizer")
VECTORIZER_PATH = get_latest_artifact(MODELS_DIR, "multinomial_nb_vectorizer")

model = None
vectorizer = None

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    model_used: str
    vectorizer_used: str

# --- Events ---
@app.on_event("startup")
def load_model():
    """Load the model and vectorizer from disk when the app starts."""
    global model, vectorizer
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model: {os.path.basename(MODEL_PATH)}")
    else:
        print("Warning: Model not found.")

    if VECTORIZER_PATH and os.path.exists(VECTORIZER_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
        print(f"Loaded vectorizer: {os.path.basename(VECTORIZER_PATH)}")
    else:
        print("Warning: Vectorizer not found.")

# --- Routes ---
@app.get('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok'}

@app.get('/model')
def model_info():
    """Returns information about the currently loaded model."""
    return {
        'model': os.path.basename(MODEL_PATH) if MODEL_PATH else None,
        'vectorizer': os.path.basename(VECTORIZER_PATH) if VECTORIZER_PATH else None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predicts the category of a given text.
    - **text**: The input text to classify.
    """
    if not model or not vectorizer:
        return {"error": "Model or vectorizer not loaded."}

    # 1. Vectorize the input text
    text_vector = vectorizer.transform([request.text])

    # 2. Make a prediction
    prediction = model.predict(text_vector)

    # 3. Return the response
    return PredictionResponse(
        prediction=str(prediction[0]),
        model_used=os.path.basename(MODEL_PATH),
        vectorizer_used=os.path.basename(VECTORIZER_PATH)
    )