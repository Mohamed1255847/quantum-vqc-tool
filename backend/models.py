from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class EncodingType(str, Enum):
    BASIS = "basis"
    AMPLITUDE = "amplitude"

class OptimizerType(str, Enum):
    NESTEROV = "nesterov"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"

class LossType(str, Enum):
    SQUARE = "square"
    CROSS_ENTROPY = "cross_entropy"
    BOTH = "both"

class DataPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    features: List[float]
    label: float

class CircuitConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    num_qubits: float = 4
    num_layers: float = 2
    encoding: EncodingType = EncodingType.BASIS

class TrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    optimizer: OptimizerType = OptimizerType.NESTEROV
    loss_type: LossType = LossType.SQUARE
    learning_rate: float = 0.5
    batch_size: float = 5
    max_iterations: float = 100
    seed: Optional[float] = 0

class TrainingRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset: List[DataPoint]
    circuit: CircuitConfig = CircuitConfig()
    training: TrainingConfig = TrainingConfig()

class TrainingResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    weights: Any
    bias: float
    training_history: List[Dict[str, Any]]
    final_cost: float
    final_cross_entropy_cost: Optional[float] = None
    final_accuracy: float
    test_accuracy: Optional[float] = None

class PredictionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    weights: Any
    bias: float
    features: List[List[float]]
    circuit: CircuitConfig = CircuitConfig()
    encoding: EncodingType = EncodingType.BASIS

class PredictionResponse(BaseModel):
    predictions: List[float]
    classes: List[int]

class RawPredictionResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    raw_values: List[float]
    classes: List[int]