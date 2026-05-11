import { useState, useCallback } from 'react';
import type {
  DataPoint,
  CircuitConfig,
  TrainingConfig,
  TrainingResponse,
  TrainingHistoryItem
} from '@/types';
import { api } from '@/lib/api';

const PARITY_SAMPLE: DataPoint[] = [
  { features: [0, 0, 0, 1], label: 1 },
  { features: [0, 0, 1, 0], label: 1 },
  { features: [0, 1, 0, 0], label: 1 },
  { features: [0, 1, 0, 1], label: 0 },
  { features: [0, 1, 1, 0], label: 0 },
  { features: [0, 1, 1, 1], label: 1 },
  { features: [1, 0, 0, 0], label: 1 },
  { features: [1, 0, 0, 1], label: 0 },
  { features: [1, 0, 1, 0], label: 0 },
  { features: [1, 0, 1, 1], label: 1 },
  { features: [1, 1, 0, 0], label: 0 },
  { features: [1, 1, 0, 1], label: 1 },
  { features: [1, 1, 1, 0], label: 1 },
  { features: [1, 1, 1, 1], label: 0 },
  { features: [0, 0, 0, 0], label: 0 },
  { features: [0, 0, 1, 1], label: 0 },
];

const IRIS_SAMPLE: DataPoint[] = [
  { features: [0.4, 0.7], label: 1 },
  { features: [0.4, 0.75], label: 1 },
  { features: [0.45, 0.8], label: 1 },
  { features: [0.5, 0.6], label: 1 },
  { features: [0.55, 0.65], label: 1 },
  { features: [0.6, 0.55], label: 0 },
  { features: [0.65, 0.5], label: 0 },
  { features: [0.7, 0.45], label: 0 },
  { features: [0.75, 0.4], label: 0 },
  { features: [0.8, 0.35], label: 0 },
  { features: [0.85, 0.3], label: 0 },
  { features: [0.9, 0.25], label: 0 },
];

interface UseTrainingReturn {
  isTraining: boolean;
  trainingHistory: TrainingHistoryItem[];
  trainingResult: TrainingResponse | null;
  train: (dataset: DataPoint[], circuitConfig: CircuitConfig, trainingConfig: TrainingConfig) => Promise<void>;
  loadSampleData: (type: 'parity' | 'iris') => DataPoint[];
  clearTraining: () => void;
  saveModel: (name: string, description: string) => Promise<void>;
  predict: (features: number[][]) => Promise<{ predictions: number[]; classes: number[] }>;
}

export function useTraining(): UseTrainingReturn {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryItem[]>([]);
  const [trainingResult, setTrainingResult] = useState<TrainingResponse | null>(null);

  const train = useCallback(async (
    dataset: DataPoint[],
    circuitConfig: CircuitConfig,
    trainingConfig: TrainingConfig
  ) => {
    setIsTraining(true);
    setTrainingHistory([]);

    try {
      const response = await api.train(dataset, circuitConfig, trainingConfig);
      setTrainingResult(response);
      setTrainingHistory(response.training_history);
    } catch (error) {
      console.error('Training failed:', error);
      alert('Training failed. Please ensure the backend is running.');
    } finally {
      setIsTraining(false);
    }
  }, []);

  const loadSampleData = useCallback((type: 'parity' | 'iris'): DataPoint[] => {
    return type === 'parity' ? [...PARITY_SAMPLE] : [...IRIS_SAMPLE];
  }, []);

  const clearTraining = useCallback(() => {
    setTrainingHistory([]);
    setTrainingResult(null);
  }, []);

  const saveModel = useCallback(async (name: string, description: string) => {
    if (!trainingResult) return;
    
    try {
      await api.saveModel(name, {
        name,
        description,
        created_at: new Date().toISOString(),
        encoding: trainingResult.weights.length > 0 ? 'varies' : 'unknown',
        final_accuracy: trainingResult.final_accuracy
      }, trainingResult.weights, trainingResult.bias);
    } catch (error) {
      console.error('Failed to save model:', error);
      alert('Failed to save model.');
    }
  }, [trainingResult]);

  const predict = useCallback(async (features: number[][]) => {
    if (!trainingResult) {
      throw new Error('No trained model available');
    }

    try {
      return await api.predict(
        trainingResult.weights,
        trainingResult.bias,
        features
      );
    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  }, [trainingResult]);

  return {
    isTraining,
    trainingHistory,
    trainingResult,
    train,
    loadSampleData,
    clearTraining,
    saveModel,
    predict
  };
}
