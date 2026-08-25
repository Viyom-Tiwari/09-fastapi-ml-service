from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

_data = load_iris()
_model = LogisticRegression(max_iter=1000).fit(_data.data, _data.target)
app = FastAPI(title="ML Prediction Service", version="1.0.0")

class Features(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/health")
def health():
    return {"status": "ok", "model": "iris-logistic-regression"}

@app.post("/predict")
def predict(x: Features):
    row = [[x.sepal_length, x.sepal_width, x.petal_length, x.petal_width]]
    probabilities = _model.predict_proba(row)[0]
    class_id = int(probabilities.argmax())
    return {
        "class_id": class_id,
        "class_name": _data.target_names[class_id],
        "confidence": round(float(probabilities[class_id]), 4),
    }
