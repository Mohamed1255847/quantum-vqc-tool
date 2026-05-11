import { useState, useEffect, useCallback } from 'react';
import { Save, Trash2, Upload, FolderOpen, CheckCircle, X, BarChart3, TrendingDown, Target } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { AlertCircle } from 'lucide-react';
import { TrainingVisualization } from '@/components/TrainingVisualization';
import type { TrainingHistoryItem } from '@/types';

interface ModelManagerProps {
  trainingResult: {
    weights: number[][][][];
    bias: number;
  } | null;
  trainingHistory: TrainingHistoryItem[];
  onTrainingComplete?: () => void;
}

interface SavedModel {
  id: string;
  metadata?: {
    name?: string;
    description?: string;
    created_at?: string;
    encoding?: string;
    num_qubits?: number;
    num_layers?: number;
    final_accuracy?: number;
    [key: string]: unknown;
  };
}

export function ModelManager({ trainingResult, trainingHistory, onTrainingComplete }: ModelManagerProps) {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [modelName, setModelName] = useState('');
  const [modelDescription, setModelDescription] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [loadedModel, setLoadedModel] = useState<{
    id: string;
    weights: number[][][][];
    bias: number;
    metadata: Record<string, unknown>;
    training_history?: TrainingHistoryItem[];
  } | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models || []);
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
      setError('Failed to load models list');
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleSaveModel = async () => {
    if (!trainingResult || !modelName.trim()) return;

    setIsSaving(true);
    setError(null);
    setSaveSuccess(false);

    try {
      const response = await fetch(`http://localhost:8000/models/save/${encodeURIComponent(modelName)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          weights: trainingResult.weights,
          bias: trainingResult.bias,
          metadata: {
            name: modelName,
            description: modelDescription,
            created_at: new Date().toISOString(),
            model_type: 'variational_classifier'
          },
          training_history: trainingHistory
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to save' }));
        throw new Error(errorData.detail || 'Failed to save model');
      }

      setModelName('');
      setModelDescription('');
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
      fetchModels();
    } catch (err) {
      console.error('Failed to save model:', err);
      setError(err instanceof Error ? err.message : 'Failed to save model');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    if (!confirm(`Are you sure you want to delete model "${modelId}"?`)) return;
    
    try {
      const response = await fetch(`http://localhost:8000/models/${encodeURIComponent(modelId)}`, { 
        method: 'DELETE' 
      });
      
      if (response.ok) {
        fetchModels();
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to delete' }));
        throw new Error(errorData.detail || 'Failed to delete model');
      }
    } catch (err) {
      console.error('Failed to delete model:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete model');
    }
  };

  const handleLoadModel = async (modelId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/models/${encodeURIComponent(modelId)}`);
      
      if (response.ok) {
        const model = await response.json();
        setLoadedModel({
          id: modelId,
          weights: model.weights,
          bias: model.bias,
          metadata: model.metadata,
          training_history: model.training_history
        });
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to load' }));
        throw new Error(errorData.detail || 'Failed to load model');
      }
    } catch (err) {
      console.error('Failed to load model:', err);
      setError(err instanceof Error ? err.message : 'Failed to load model');
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center gap-2 text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span>{error}</span>
          <Button variant="ghost" size="sm" onClick={() => setError(null)} className="ml-auto">Dismiss</Button>
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Save className="h-5 w-5" />
            Save Current Model
          </CardTitle>
          <CardDescription>
            Save the trained model weights and configuration
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!trainingResult ? (
            <div className="p-4 bg-muted/50 rounded-lg text-center">
              <p className="text-sm text-muted-foreground">
                Train a model first before saving.
              </p>
              {onTrainingComplete && (
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="mt-3"
                  onClick={onTrainingComplete}
                >
                  Go to Training
                </Button>
              )}
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Model Name</Label>
                  <Input
                    placeholder="my-quantum-model"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Description (optional)</Label>
                  <Input
                    placeholder="Parity classifier for binary data"
                    value={modelDescription}
                    onChange={(e) => setModelDescription(e.target.value)}
                  />
                </div>
              </div>
              <Button 
                onClick={handleSaveModel} 
                disabled={!modelName.trim() || isSaving} 
                className="gap-2"
              >
                <Save className="h-4 w-4" />
                {isSaving ? 'Saving...' : 'Save Model'}
              </Button>
              {saveSuccess && (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="h-4 w-4" />
                  <span className="text-sm">Model saved successfully!</span>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {loadedModel && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Loaded Model: {loadedModel.id}
              </CardTitle>
              <Button variant="ghost" size="sm" onClick={() => setLoadedModel(null)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            {loadedModel.metadata && (
              <CardDescription>
                {loadedModel.metadata.description as string}
                {loadedModel.metadata.created_at && (
                  <span> &middot; Created: {new Date(loadedModel.metadata.created_at as string).toLocaleDateString()}</span>
                )}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <Card className="bg-muted/50">
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <TrendingDown className="h-4 w-4" />
                    Final Cost
                  </div>
                  <div className="text-2xl font-bold mt-1">
                    {loadedModel.training_history
                      ? loadedModel.training_history[loadedModel.training_history.length - 1].cost.toFixed(6)
                      : loadedModel.metadata.final_cost?.toFixed(6) ?? 'N/A'}
                  </div>
                </CardContent>
              </Card>
              <Card className="bg-muted/50">
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Target className="h-4 w-4" />
                    Final Accuracy
                  </div>
                  <div className="text-2xl font-bold mt-1">
                    {loadedModel.training_history
                      ? `${(loadedModel.training_history[loadedModel.training_history.length - 1].accuracy * 100).toFixed(1)}%`
                      : loadedModel.metadata.final_accuracy
                        ? `${(loadedModel.metadata.final_accuracy as number * 100).toFixed(1)}%`
                        : 'N/A'}
                  </div>
                </CardContent>
              </Card>
              <Card className="bg-muted/50">
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <BarChart3 className="h-4 w-4" />
                    Iterations
                  </div>
                  <div className="text-2xl font-bold mt-1">
                    {loadedModel.training_history?.length ?? 'N/A'}
                  </div>
                </CardContent>
              </Card>
            </div>
            {loadedModel.training_history && loadedModel.training_history.length > 0 && (
              <TrainingVisualization data={loadedModel.training_history} />
            )}
            {!loadedModel.training_history && (
              <div className="p-4 bg-muted/30 rounded-lg text-center text-sm text-muted-foreground">
                Full training history is not available for this model. Only summary stats are shown.
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FolderOpen className="h-5 w-5" />
            Saved Models
          </CardTitle>
          <CardDescription>
            Load or delete previously saved models
          </CardDescription>
        </CardHeader>
        <CardContent>
          {models.length === 0 ? (
            <div className="text-center py-8">
              <FolderOpen className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">
                No saved models yet.
              </p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                Train a model and save it to see it here.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {models.map((model) => (
                <div
                  key={model.id}
                  className="flex items-center justify-between p-4 rounded-lg border hover:bg-muted/30 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium truncate">{model.id}</h4>
                    {model.metadata && (
                      <div className="flex flex-wrap gap-2 mt-1 text-xs text-muted-foreground">
                        {model.metadata.created_at && (
                          <span>Created: {new Date(model.metadata.created_at).toLocaleDateString()}</span>
                        )}
                        {model.metadata.description && (
                          <span className="truncate">{model.metadata.description}</span>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="flex gap-2 ml-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleLoadModel(model.id)}
                      className="gap-1"
                    >
                      <Upload className="h-3 w-3" />
                      Load
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => handleDeleteModel(model.id)}
                      className="gap-1"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}