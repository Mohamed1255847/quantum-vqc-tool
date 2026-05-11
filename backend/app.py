from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import numpy as np

from models import (
    TrainingRequest, TrainingResponse, PredictionRequest, PredictionResponse
)
from vqc import (
    train_variational_classifier, predict, save_model, load_model, list_models
)

app = FastAPI(title="Quantum Variational Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SaveModelRequest(BaseModel):
    weights: Any
    bias: float
    metadata: Dict[str, Any]
    training_history: Optional[List[Dict[str, Any]]] = None

@app.get("/")
async def root():
    return {"message": "Quantum Variational Classifier API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/train", response_model=TrainingResponse)
async def train(request: TrainingRequest):
    try:
        X = [dp.features for dp in request.dataset]
        Y = [float(dp.label) for dp in request.dataset]
        
        encoding = str(request.circuit.encoding.value if hasattr(request.circuit.encoding, 'value') else request.circuit.encoding)
        optimizer = str(request.training.optimizer.value if hasattr(request.training.optimizer, 'value') else request.training.optimizer)
        loss_type = str(request.training.loss_type.value if hasattr(request.training.loss_type, 'value') else request.training.loss_type)
        
        num_qubits = int(request.circuit.num_qubits)
        num_layers = int(request.circuit.num_layers)
        batch_size = int(request.training.batch_size)
        max_iterations = int(request.training.max_iterations)
        seed = int(request.training.seed) if request.training.seed else None
        learning_rate = float(request.training.learning_rate)
        
        weights, bias, history = train_variational_classifier(
            X=X,
            Y=Y,
            num_qubits=num_qubits,
            num_layers=num_layers,
            encoding=encoding,
            optimizer_type=optimizer,
            loss_type=loss_type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_iterations=max_iterations,
            seed=seed
        )
        
        # Convert weights to proper nested list (weights shape: [num_layers, num_qubits, 3])
        weights_array = np.array(weights)
        weights_list = weights_array.tolist()
        
        return TrainingResponse(
            weights=weights_list,
            bias=float(bias),
            training_history=history,
            final_cost=history[-1]["cost"] if history else 0,
            final_cross_entropy_cost=history[-1].get("cost_cross_entropy"),
            final_accuracy=history[-1]["accuracy"] if history else 0,
            test_accuracy=history[-1]["accuracy"] if history else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    try:
        encoding = request.encoding.value if hasattr(request.encoding, 'value') else str(request.encoding)
        
        predictions, classes = predict(
            weights=request.weights,
            bias=request.bias,
            X=request.features,
            encoding=encoding,
            num_qubits=request.circuit.num_qubits
        )
        return PredictionResponse(predictions=predictions, classes=classes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/save/{model_id}")
async def save_model_endpoint(model_id: str, request: SaveModelRequest):
    try:
        save_model(model_id, request.weights, request.bias, request.metadata, request.training_history)
        return {"status": "success", "message": f"Model {model_id} saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    models = list_models()
    return {"models": models}

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    model_path = os.path.join("models", f"{model_id}.json")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    os.remove(model_path)
    return {"status": "success", "message": f"Model {model_id} deleted"}

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    model = load_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@app.get("/preprocessing-info")
async def get_preprocessing_info():
    return {
        "basis_encoding": {
            "description": "Encode binary inputs as computational basis states",
            "input_type": "binary strings (0s and 1s)",
            "example": [[0, 1, 0, 1], [1, 0, 0, 0]]
        },
        "amplitude_encoding": {
            "description": "Encode real vectors as amplitude vectors",
            "input_type": "real-valued vectors",
            "preprocessing": ["padding", "normalization", "angle extraction"],
            "example": [[0.4, 0.75], [0.5, 0.5]]
        }
    }

@app.get("/default-configs")
async def get_default_configs():
    return {
        "parity_classification": {
            "circuit": {"num_qubits": 4, "num_layers": 2},
            "training": {
                "optimizer": "nesterov",
                "learning_rate": 0.5,
                "batch_size": 5,
                "max_iterations": 100
            },
            "encoding": "basis"
        },
        "iris_classification": {
            "circuit": {"num_qubits": 2, "num_layers": 6},
            "training": {
                "optimizer": "nesterov",
                "learning_rate": 0.01,
                "batch_size": 5,
                "max_iterations": 60
            },
            "encoding": "amplitude"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
