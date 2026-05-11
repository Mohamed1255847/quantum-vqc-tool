import type {
  DataPoint,
  CircuitConfig,
  TrainingConfig,
  TrainingResponse,
  PredictionResponse
} from '@/types';

const API_BASE = 'http://localhost:8000';

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  async train(
    dataset: DataPoint[],
    circuit: CircuitConfig,
    training: TrainingConfig
  ): Promise<TrainingResponse> {
    return request<TrainingResponse>('/train', {
      method: 'POST',
      body: JSON.stringify({ dataset, circuit, training }),
    });
  },

  async predict(
    weights: number[][][][],
    bias: number,
    features: number[][],
    circuit: CircuitConfig = { num_qubits: 4, num_layers: 2, encoding: 'basis' },
    encoding: 'basis' | 'amplitude' = 'basis'
  ): Promise<PredictionResponse> {
    return request<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify({ weights, bias, features, circuit, encoding }),
    });
  },

  async saveModel(
    modelId: string,
    metadata: Record<string, unknown>,
    weights: number[][][][],
    bias: number
  ): Promise<void> {
    await request(`/models/save/${modelId}`, {
      method: 'POST',
      body: JSON.stringify({ weights, bias, metadata }),
    });
  },

  async getModels(): Promise<{ models: Array<{ id: string; metadata: Record<string, unknown> }> }> {
    return request('/models');
  },

  async getModel(modelId: string): Promise<{
    weights: number[][][][];
    bias: number;
    metadata: Record<string, unknown>;
  }> {
    return request(`/models/${modelId}`);
  },

  async getPreprocessingInfo(): Promise<Record<string, unknown>> {
    return request('/preprocessing-info');
  },

  async getDefaultConfigs(): Promise<Record<string, unknown>> {
    return request('/default-configs');
  },

  async health(): Promise<{ status: string }> {
    return request('/health');
  }
};
