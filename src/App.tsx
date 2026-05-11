import { useState, useCallback, useEffect } from 'react';
import { Brain, Upload, Settings, Save, Play, RotateCcw, AlertCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { DataUpload } from '@/components/DataUpload';
import { CircuitConfig } from '@/components/CircuitConfig';
import { TrainingVisualization } from '@/components/TrainingVisualization';
import { ModelManager } from '@/components/ModelManager';
import type { DataPoint, CircuitConfig as CircuitConfigType, TrainingConfig, TrainingHistoryItem } from '@/types';

type Tab = 'data' | 'circuit' | 'training' | 'models';

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

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('data');
  const [dataset, setDataset] = useState<DataPoint[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryItem[]>([]);
  const [trainingResult, setTrainingResult] = useState<{
    weights: number[][][][];
    bias: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendConnected, setBackendConnected] = useState(false);
  const [showTrainingResults, setShowTrainingResults] = useState(false);

  const [circuitConfig, setCircuitConfig] = useState<CircuitConfigType>({
    num_qubits: 4,
    num_layers: 2,
    encoding: 'basis'
  });
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    optimizer: 'nesterov',
    loss_type: 'square',
    learning_rate: 0.5,
    batch_size: 5,
    max_iterations: 100,
    seed: 0
  });

  const tabs = [
    { id: 'data' as const, label: 'Data Upload', icon: Upload },
    { id: 'circuit' as const, label: 'Circuit Config', icon: Settings },
    { id: 'training' as const, label: 'Training', icon: Play },
    { id: 'models' as const, label: 'Saved Models', icon: Save },
  ];

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch('http://localhost:8000/health');
        if (res.ok) {
          setBackendConnected(true);
          setError(null);
        }
      } catch {
        setBackendConnected(false);
        setError('Cannot connect to backend. Please start: cd backend && python app.py');
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleDatasetChange = useCallback((newDataset: DataPoint[]) => {
    setDataset(newDataset);
    setTrainingHistory([]);
    setTrainingResult(null);
    setShowTrainingResults(false);
  }, []);

  const loadSampleData = useCallback((type: 'parity' | 'iris') => {
    const data = type === 'parity' ? [...PARITY_SAMPLE] : [...IRIS_SAMPLE];
    setDataset(data);
    setCircuitConfig(prev => ({
      ...prev,
      encoding: type === 'parity' ? 'basis' : 'amplitude',
      num_qubits: type === 'parity' ? 4 : 2,
      num_layers: type === 'parity' ? 2 : 6
    }));
    setTrainingConfig(prev => ({
      ...prev,
      max_iterations: type === 'parity' ? 100 : 60,
      learning_rate: type === 'parity' ? 0.5 : 0.01
    }));
    setTrainingHistory([]);
    setTrainingResult(null);
    setShowTrainingResults(false);
  }, []);

  const clearTraining = useCallback(() => {
    setTrainingHistory([]);
    setTrainingResult(null);
    setError(null);
    setShowTrainingResults(false);
  }, []);

  const handleStartTraining = async () => {
    if (dataset.length === 0) {
      setError('Please upload or generate a dataset first');
      return;
    }

    setIsTraining(true);
    setError(null);
    setShowTrainingResults(false);

    try {
      const response = await fetch('http://localhost:8000/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset: dataset,
          circuit: circuitConfig,
          training: trainingConfig
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      setTrainingResult({ weights: result.weights, bias: result.bias });
      setTrainingHistory(result.training_history || []);
      setShowTrainingResults(true);
      setActiveTab('training');
    } catch (err) {
      console.error('Training failed:', err);
      setError(`Training failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsTraining(false);
    }
  };

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto flex items-center justify-between px-4 py-4">
          <div className="flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            <h1 className="text-2xl font-bold">Quantum VQC Tool</h1>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${backendConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <p className="text-sm text-muted-foreground">
              {backendConnected ? 'Backend connected' : 'Backend offline'}
            </p>
          </div>
        </div>
      </header>

      {error && (
        <div className="mx-4 mt-4 p-4 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center gap-2 text-destructive">
          <AlertCircle className="h-5 w-5" />
          <span>{error}</span>
          <Button variant="ghost" size="sm" onClick={() => setError(null)} className="ml-auto">Dismiss</Button>
        </div>
      )}

      <div className="container mx-auto px-4 py-6">
        <div className="flex gap-2 mb-6 flex-wrap">
          {tabs.map(({ id, label, icon: Icon }) => (
            <Button
              key={id}
              variant={activeTab === id ? 'default' : 'outline'}
              onClick={() => setActiveTab(id)}
              className="gap-2"
            >
              <Icon className="h-4 w-4" />
              {label}
            </Button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            {activeTab === 'data' && (
              <DataUpload
                dataset={dataset}
                onDatasetChange={handleDatasetChange}
                onLoadSample={loadSampleData}
              />
            )}

            {activeTab === 'circuit' && (
              <CircuitConfig
                config={circuitConfig}
                onChange={setCircuitConfig}
              />
            )}

            {activeTab === 'training' && (
              <Card>
                <CardHeader>
                  <CardTitle>Training Configuration</CardTitle>
                  <CardDescription>
                    Configure and run the variational quantum classifier training
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Optimizer</Label>
                        <select
                          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                          value={trainingConfig.optimizer}
                          onChange={(e) => setTrainingConfig(prev => ({
                            ...prev,
                            optimizer: e.target.value as TrainingConfig['optimizer']
                          }))}
                        >
                          <option value="nesterov">Nesterov Momentum</option>
                          <option value="adam">Adam</option>
                          <option value="rmsprop">RMSProp</option>
                          <option value="gradient_descent">Gradient Descent</option>
                        </select>
                      </div>
                      <div className="space-y-2">
                        <Label>Loss Function</Label>
                        <select
                          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                          value={trainingConfig.loss_type}
                          onChange={(e) => setTrainingConfig(prev => ({
                            ...prev,
                            loss_type: e.target.value as TrainingConfig['loss_type']
                          }))}
                        >
                          <option value="square">Square Loss</option>
                          <option value="cross_entropy">Cross-Entropy</option>
                          <option value="both">Both (Square + Cross-Ent)</option>
                        </select>
                      </div>
                      <div className="space-y-2">
                        <Label>Batch Size</Label>
                        <Input
                          type="number"
                          value={trainingConfig.batch_size}
                          onChange={(e) => setTrainingConfig(prev => ({
                            ...prev,
                            batch_size: parseInt(e.target.value) || 5
                          }))}
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label>Learning Rate: {trainingConfig.learning_rate.toFixed(3)}</Label>
                      <Slider
                        value={[trainingConfig.learning_rate]}
                        min={0.001}
                        max={1}
                        step={0.001}
                        onValueChange={(value) => setTrainingConfig(prev => ({
                          ...prev,
                          learning_rate: value[0]
                        }))}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Max Iterations: {trainingConfig.max_iterations}</Label>
                      <Slider
                        value={[trainingConfig.max_iterations]}
                        min={10}
                        max={500}
                        step={10}
                        onValueChange={(value) => setTrainingConfig(prev => ({
                          ...prev,
                          max_iterations: value[0]
                        }))}
                      />
                    </div>

                    <div className="flex gap-2">
                      <Button
                        onClick={handleStartTraining}
                        disabled={isTraining || dataset.length === 0 || !backendConnected}
                        className="gap-2"
                      >
                        {isTraining ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Training...
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4" />
                            Start Training
                          </>
                        )}
                      </Button>
                      <Button
                        variant="outline"
                        onClick={clearTraining}
                        className="gap-2"
                      >
                        <RotateCcw className="h-4 w-4" />
                        Clear Results
                      </Button>
                    </div>
                  </div>

                  {(showTrainingResults && trainingHistory.length > 0) && (
                    <TrainingVisualization data={trainingHistory} />
                  )}
                </CardContent>
              </Card>
            )}

            {activeTab === 'models' && (
              <ModelManager trainingResult={trainingResult} trainingHistory={trainingHistory} onTrainingComplete={() => {
                setShowTrainingResults(true);
                setActiveTab('training');
              }} />
            )}
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Dataset Info</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Samples:</span>
                    <span className="font-medium">{dataset.length}</span>
                  </div>
                  {dataset.length > 0 && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Features:</span>
                        <span className="font-medium">{dataset[0].features.length}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Encoding:</span>
                        <span className="font-medium">{circuitConfig.encoding}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Classes:</span>
                        <span className="font-medium">
                          {[...new Set(dataset.map(d => d.label))].join(', ')}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Circuit Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Qubits:</span>
                    <span className="font-medium">{circuitConfig.num_qubits}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Layers:</span>
                    <span className="font-medium">{circuitConfig.num_layers}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Parameters:</span>
                    <span className="font-medium">
                      {circuitConfig.num_layers * circuitConfig.num_qubits * 3}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {showTrainingResults && trainingHistory.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Quick Stats</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Final Cost:</span>
                    <span className="font-mono text-sm">{trainingHistory[trainingHistory.length - 1].cost.toFixed(6)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Final Accuracy:</span>
                    <span className="font-mono text-sm">{(trainingHistory[trainingHistory.length - 1].accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={(trainingHistory.length / trainingConfig.max_iterations) * 100} className="h-2" />
                  <p className="text-xs text-muted-foreground text-center">
                    {trainingHistory.length} / {trainingConfig.max_iterations} iterations
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;