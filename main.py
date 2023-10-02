from fastapi import FastAPI
from test import load_model, CLASSES
from preprocess import preprocess_message

app = FastAPI()

model = load_model()

@app.post("/predict")
async def predict(message: str):
    # Preprocess the message
    processed_message = preprocess_message(message)

    X = []
    X.append(processed_message)

    # Make a prediction
    prediction = model.predict(X)

    probabilites = model.predict_proba(X)
    predictions = (probabilites[:,1] >= 0.85).astype(int)

    # Return the prediction
    return {"prediction": str(CLASSES[predictions[0]])}
