export interface DataPoint {
  features: number[];
  label: number;
}

export interface CircuitConfig {
  num_qubits: number;
  num_layers: number;
  encoding: 'basis' | 'amplitude';
}

export interface TrainingConfig {
  optimizer: 'nesterov' | 'adam' | 'rmsprop' | 'gradient_descent';
  loss_type: 'square' | 'cross_entropy' | 'both';
  learning_rate: number;
  batch_size: number;
  max_iterations: number;
  seed: number | null;
}

export interface TrainingRequest {
  dataset: DataPoint[];
  circuit: CircuitConfig;
  training: TrainingConfig;
}

export interface TrainingHistoryItem {
  iteration: number;
  cost: number;
  cost_cross_entropy?: number;
  accuracy: number;
}

export interface TrainingResponse {
  weights: number[][][][];
  bias: number;
  training_history: TrainingHistoryItem[];
  final_cost: number;
  final_cross_entropy_cost?: number;
  final_accuracy: number;
  test_accuracy: number | null;
}

export interface PredictionRequest {
  weights: number[][][][];
  bias: number;
  features: number[][];
  circuit: CircuitConfig;
  encoding: 'basis' | 'amplitude';
}

export interface PredictionResponse {
  predictions: number[];
  classes: number[];
}

export interface SavedModel {
  id: string;
  metadata: {
    name?: string;
    description?: string;
    created_at?: string;
    encoding?: string;
    num_qubits?: number;
    num_layers?: number;
    final_accuracy?: number;
  };
}
