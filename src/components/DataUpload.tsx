import { useState, useCallback, useEffect } from 'react';
import { Upload as UploadIcon, FileText, Trash2, Sparkles, Database } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { DataPoint } from '@/types';

interface DataUploadProps {
  dataset: DataPoint[];
  onDatasetChange: (dataset: DataPoint[]) => void;
  onLoadSample: (type: 'parity' | 'iris') => void;
}

export function DataUpload({ dataset, onDatasetChange, onLoadSample }: DataUploadProps) {
  const [csvInput, setCsvInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [largeSize, setLargeSize] = useState(10000);
  const [largeFeatures, setLargeFeatures] = useState(8);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    setError(null);
  }, [dataset]);

  const parseCSV = useCallback((text: string): DataPoint[] => {
    const lines = text.trim().split('\n').filter(line => line.trim());
    if (lines.length === 0) return [];
    const startIndex = lines[0].split(',').some(v => isNaN(parseFloat(v.trim()))) ? 1 : 0;
    return lines.slice(startIndex).map(line => {
      const parts = line.split(',').map(p => p.trim());
      return { features: parts.slice(0, -1).map(parseFloat), label: parseInt(parts[parts.length - 1], 10) };
    });
  }, []);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const parsed = parseCSV(e.target?.result as string || '');
        if (parsed.length === 0 || parsed.some(p => isNaN(p.label) || p.features.some(isNaN)))
          throw new Error('Invalid data format');
        onDatasetChange(parsed);
        setError(null);
      } catch (err) {
        setError('Failed to parse CSV. Ensure format is: feature1,feature2,...,label');
      }
    };
    reader.readAsText(file);
  }, [onDatasetChange, parseCSV]);

  const handleCsvSubmit = useCallback(() => {
    if (!csvInput.trim()) return;
    try {
      const parsed = parseCSV(csvInput);
      if (parsed.length === 0 || parsed.some(p => isNaN(p.label) || p.features.some(isNaN)))
        throw new Error('Invalid data format');
      onDatasetChange(parsed);
      setError(null);
    } catch (err) {
      setError('Failed to parse CSV. Ensure format is: feature1,feature2,...,label');
    }
  }, [csvInput, onDatasetChange, parseCSV]);

  const generateLargeDataset = useCallback(() => {
    setGenerating(true);
    setError(null);

    setTimeout(() => {
      try {
        const data: DataPoint[] = [];
        for (let i = 0; i < largeSize; i++) {
          const features = Array.from({ length: largeFeatures }, () =>
            Math.round(Math.random())
          );
          const label = features.reduce((a, b) => a + b, 0) % 2;
          data.push({ features, label });
        }
        onDatasetChange(data);
        setError(null);
      } catch (err) {
        setError('Failed to generate dataset');
      } finally {
        setGenerating(false);
      }
    }, 0);
  }, [largeSize, largeFeatures, onDatasetChange]);

  const handleClear = useCallback(() => {
    onDatasetChange([]);
    setCsvInput('');
    setError(null);
  }, [onDatasetChange]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Generate Large Dataset
          </CardTitle>
          <CardDescription>
            Generate a synthetic dataset for performance testing
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Number of samples</Label>
              <Input
                type="number"
                min={100}
                max={100000}
                step={100}
                value={largeSize}
                onChange={(e) => setLargeSize(parseInt(e.target.value) || 10000)}
              />
            </div>
            <div className="space-y-2">
              <Label>Features per sample</Label>
              <Input
                type="number"
                min={2}
                max={20}
                step={1}
                value={largeFeatures}
                onChange={(e) => setLargeFeatures(parseInt(e.target.value) || 8)}
              />
            </div>
          </div>
          <Button
            onClick={generateLargeDataset}
            disabled={generating}
            className="gap-2 w-full"
            variant="secondary"
          >
            <Database className="h-4 w-4" />
            {generating ? 'Generating...' : `Generate ${largeSize.toLocaleString()} Samples`}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <UploadIcon className="h-5 w-5" />
            Upload Dataset
          </CardTitle>
          <CardDescription>
            Upload a CSV file or paste data directly. Format: feature1,feature2,...,label
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4 flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <Input
                type="file"
                accept=".csv,.txt"
                onChange={handleFileUpload}
                className="cursor-pointer"
              />
            </div>
            <Button
              variant="outline"
              onClick={() => onLoadSample('parity')}
              className="gap-2"
            >
              <Sparkles className="h-4 w-4" />
              Load Parity Sample
            </Button>
            <Button
              variant="outline"
              onClick={() => onLoadSample('iris')}
              className="gap-2"
            >
              <Sparkles className="h-4 w-4" />
              Load Iris Sample
            </Button>
          </div>

          <div className="border-t pt-4">
            <Label>Or paste CSV data:</Label>
            <textarea
              className="flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm mt-2 min-h-[120px] font-mono"
              placeholder="0.4,0.75,1&#10;0.5,0.6,0&#10;..."
              value={csvInput}
              onChange={(e) => setCsvInput(e.target.value)}
            />
            <div className="flex gap-2 mt-2">
              <Button onClick={handleCsvSubmit} size="sm" disabled={!csvInput.trim()}>
                Parse Data
              </Button>
              {dataset.length > 0 && (
                <Button variant="outline" onClick={handleClear} size="sm">
                  <Trash2 className="h-4 w-4 mr-1" />
                  Clear
                </Button>
              )}
            </div>
          </div>

          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}
        </CardContent>
      </Card>

      {dataset.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Data Preview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="px-2 py-2 text-left font-medium">#</th>
                    {dataset[0].features.map((_, i) => (
                      <th key={i} className="px-2 py-2 text-left font-medium">
                        Feature {i + 1}
                      </th>
                    ))}
                    <th className="px-2 py-2 text-left font-medium">Label</th>
                  </tr>
                </thead>
                <tbody>
                  {dataset.slice(0, 10).map((dp, i) => (
                    <tr key={i} className="border-b">
                      <td className="px-2 py-2 text-muted-foreground">{i + 1}</td>
                      {dp.features.map((f, j) => (
                        <td key={j} className="px-2 py-2">{f.toFixed(4)}</td>
                      ))}
                      <td className="px-2 py-2 font-medium">{dp.label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {dataset.length > 10 && (
                <p className="text-sm text-muted-foreground mt-2">
                  ... and {dataset.length - 10} more rows
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
