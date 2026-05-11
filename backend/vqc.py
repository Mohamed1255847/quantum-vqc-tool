import pennylane as qml
from pennylane import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import json
import os

MODELS_DIR = "models"

def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)

def square_loss(labels, predictions):
    return np.mean((labels - np.array(predictions)) ** 2)

def cross_entropy_loss(labels, predictions):
    eps = 1e-8
    labels = (np.array(labels) + 1) / 2
    probs = (np.array(predictions) + 1) / 2
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    return acc / len(labels)

def train_variational_classifier(
    X,
    Y,
    num_qubits: int = 4,
    num_layers: int = 2,
    encoding: str = "basis",
    optimizer_type: str = "nesterov",
    loss_type: str = "square",
    learning_rate: float = 0.5,
    batch_size: int = 5,
    max_iterations: int = 100,
    seed: Optional[int] = None
) -> Tuple:
    if seed is not None:
        np.random.seed(seed)
    
    # Convert everything to Python floats to avoid autograd issues
    X = [[float(v) for v in row] for row in X]
    Y = [float(y) for y in Y]
    
    X = np.array(X, dtype=float)
    Y_shifted = np.array(Y, dtype=float) * 2 - 1
    
    if encoding == "basis":
        num_qubits_actual = len(X[0])
    else:
        num_qubits_actual = num_qubits
    
    weights = 0.01 * np.random.randn(num_layers, num_qubits_actual, 3)
    history = []
    
    dev = qml.device("default.qubit", wires=num_qubits_actual)
    
    @qml.qnode(dev)
    def circuit(weights, x):
        for i, val in enumerate(x):
            if val > 0.5:
                qml.PauliX(wires=i)
        
        for layer_weights in weights:
            for wire in range(num_qubits_actual):
                qml.Rot(layer_weights[wire, 0], layer_weights[wire, 1], layer_weights[wire, 2], wires=wire)
            for i in range(num_qubits_actual):
                qml.CNOT(wires=[i, (i + 1) % num_qubits_actual])
        
        return qml.expval(qml.PauliZ(0))
    
    def get_loss_for_training(loss_type):
        if loss_type == "cross_entropy":
            return cross_entropy_loss
        return square_loss
    
    def cost_fn(w, X_batch, Y_batch):
        batch_preds = []
        for i in range(len(X_batch)):
            pred = circuit(w, X_batch[i])
            batch_preds.append(pred)
        loss_func = get_loss_for_training(loss_type)
        return loss_func(Y_batch, batch_preds)
    
    if optimizer_type == "nesterov":
        opt = qml.optimize.NesterovMomentumOptimizer(learning_rate)
    elif optimizer_type == "adam":
        opt = qml.optimize.AdamOptimizer(learning_rate)
    elif optimizer_type == "rmsprop":
        opt = qml.optimize.RMSPropOptimizer(learning_rate)
    else:
        opt = qml.optimize.GradientDescentOptimizer(learning_rate)
    
    for it in range(max_iterations):
        n = len(X)
        bs = min(batch_size, n)
        idx = np.random.choice(n, bs, replace=False)
        X_batch = X[idx]
        Y_batch = Y_shifted[idx]
        
        weights = opt.step(lambda w: cost_fn(w, X_batch, Y_batch), weights)
        
        all_preds = np.array([circuit(weights, x) for x in X])
        acc = accuracy(Y_shifted, np.sign(all_preds))
        
        if loss_type == "both":
            sq_loss = float(square_loss(Y_shifted, all_preds))
            ce_loss = float(cross_entropy_loss(Y_shifted, all_preds))
            history.append({
                "iteration": it + 1,
                "cost": sq_loss,
                "cost_cross_entropy": ce_loss,
                "accuracy": float(acc)
            })
            print(f"Iter: {it + 1:4d} | Square: {sq_loss:0.7f} | CrossEnt: {ce_loss:0.7f} | Acc: {acc:0.4f}")
        else:
            loss_func = get_loss_for_training(loss_type)
            current_cost = float(loss_func(Y_shifted, all_preds))
            history.append({
                "iteration": it + 1,
                "cost": current_cost,
                "accuracy": float(acc)
            })
            if (it + 1) % 10 == 0:
                print(f"Iter: {it + 1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
    
    return weights, 0.0, history

def predict(weights, bias, X, encoding="basis", num_qubits=4):
    X = [[float(v) for v in row] for row in X]
    X = np.array(X, dtype=float)
    
    if encoding == "basis":
        num_qubits_actual = len(X[0])
    else:
        num_qubits_actual = num_qubits
    
    dev = qml.device("default.qubit", wires=num_qubits_actual)
    
    @qml.qnode(dev)
    def circuit(weights, x):
        for i, val in enumerate(x):
            if val > 0.5:
                qml.PauliX(wires=i)
        
        for layer_weights in weights:
            for wire in range(num_qubits_actual):
                qml.Rot(layer_weights[wire, 0], layer_weights[wire, 1], layer_weights[wire, 2], wires=wire)
            for i in range(num_qubits_actual):
                qml.CNOT(wires=[i, (i + 1) % num_qubits_actual])
        
        return qml.expval(qml.PauliZ(0))
    
    predictions = []
    classes = []
    for x in X:
        pred = float(np.sign(circuit(weights, x) + bias))
        predictions.append(pred)
        classes.append(int(pred))
    
    return predictions, classes

def save_model(model_id: str, weights, bias: float, metadata: Dict[str, Any], training_history: Optional[List[Dict[str, Any]]] = None):
    ensure_models_dir()
    model_data = {
        "weights": weights.tolist() if hasattr(weights, "tolist") else weights,
        "bias": bias,
        "metadata": metadata,
    }
    if training_history:
        model_data["training_history"] = training_history
    with open(os.path.join(MODELS_DIR, f"{model_id}.json"), "w") as f:
        json.dump(model_data, f)
    return True

def load_model(model_id: str) -> Optional[Dict[str, Any]]:
    model_path = os.path.join(MODELS_DIR, f"{model_id}.json")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "r") as f:
        return json.load(f)

def list_models() -> List[Dict[str, Any]]:
    ensure_models_dir()
    models = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(MODELS_DIR, filename), "r") as f:
                model_data = json.load(f)
                models.append({
                    "id": filename.replace(".json", ""),
                    "metadata": model_data.get("metadata", {})
                })
    return models
