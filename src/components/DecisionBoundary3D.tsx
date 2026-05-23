import { useState, useRef, useEffect, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { AlertCircle, CuboidIcon as Cube3d } from 'lucide-react';
import type { DataPoint, CircuitConfig } from '@/types';

interface DecisionBoundary3DProps {
  dataset: DataPoint[];
  weights: number[][][][];
  bias: number;
  circuitConfig: CircuitConfig;
}

const API_BASE = 'http://localhost:8000';
const GRID_RESOLUTION = 40;

declare global {
  interface Window {
    Plotly: any;
  }
}

export function DecisionBoundary3D({ dataset, weights, bias, circuitConfig }: DecisionBoundary3DProps) {
  const [plotReady, setPlotReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [featIdx0, setFeatIdx0] = useState(0);
  const [featIdx1, setFeatIdx1] = useState(1);
  const plotRef = useRef<HTMLDivElement>(null);

  const numFeatures = dataset.length > 0 ? dataset[0].features.length : 0;

  const featureLabels = useMemo(() => {
    if (numFeatures <= 2) return ['Feature 1', 'Feature 2'];
    return Array.from({ length: numFeatures }, (_, i) => `Feature ${i + 1}`);
  }, [numFeatures]);

  const fillValues = useMemo(() => {
    if (numFeatures <= 2) return [];
    return Array.from({ length: numFeatures }, (_, fi) => {
      const vals = dataset.map(d => d.features[fi]).sort((a, b) => a - b);
      const mid = Math.floor(vals.length / 2);
      return vals.length % 2 === 0 ? (vals[mid - 1] + vals[mid]) / 2 : vals[mid];
    });
  }, [dataset, numFeatures]);

  useEffect(() => {
    if (typeof window.Plotly !== 'undefined') {
      setPlotReady(true);
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-3.0.1.min.js';
    script.onload = () => setPlotReady(true);
    script.onerror = () => setError('Failed to load Plotly.js from CDN');
    document.body.appendChild(script);
    return () => { document.body.removeChild(script); };
  }, []);

  const handleGenerate = async () => {
    if (numFeatures < 2) {
      setError('Dataset must have at least 2 features');
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const sel0 = numFeatures > 2 ? featIdx0 : 0;
      const sel1 = numFeatures > 2 ? featIdx1 : 1;

      const f0 = dataset.map(d => d.features[sel0]);
      const f1 = dataset.map(d => d.features[sel1]);
      const min0 = Math.min(...f0);
      const max0 = Math.max(...f0);
      const min1 = Math.min(...f1);
      const max1 = Math.max(...f1);
      const pad0 = (max0 - min0) * 0.15 || 0.2;
      const pad1 = (max1 - min1) * 0.15 || 0.2;

      const x = linspace(min0 - pad0, max0 + pad0, GRID_RESOLUTION);
      const y = linspace(min1 - pad1, max1 + pad1, GRID_RESOLUTION);

      const gridPoints: number[][] = [];
      for (let j = 0; j < y.length; j++) {
        for (let i = 0; i < x.length; i++) {
          const point = numFeatures > 2 ? [...fillValues] : [x[i], y[j]];
          point[sel0] = x[i];
          point[sel1] = y[j];
          gridPoints.push(point);
        }
      }

      const resp = await fetch(`${API_BASE}/predict/raw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          weights,
          bias,
          features: gridPoints,
          circuit: circuitConfig,
          encoding: circuitConfig.encoding,
        }),
      });

      if (!resp.ok) throw new Error(`Prediction failed: ${resp.statusText}`);
      const data = await resp.json();
      const rawValues: number[] = data.raw_values;

      const Z: number[][] = [];
      for (let j = 0; j < y.length; j++) {
        Z.push(rawValues.slice(j * x.length, (j + 1) * x.length));
      }

      const scatterData = dataset.map(d => ({
        x: d.features[sel0],
        y: d.features[sel1],
        label: d.label,
      }));

      const traceSurface: any = {
        type: 'surface',
        x, y, z: Z,
        colorscale: [
          [0, '#3b82f6'],
          [0.5, '#f8fafc'],
          [1, '#ef4444'],
        ],
        showscale: true,
        opacity: 0.85,
        contours: {
          z: { show: true, usecolormap: true, highlightcolor: '#fbbf24', project: { z: true } },
        },
        hovertemplate: `${featureLabels[sel0]}: %{x:.3f}<br>${featureLabels[sel1]}: %{y:.3f}<br>pred: %{z:.3f}<extra></extra>`,
      };

      const traceClass0: any = {
        type: 'scatter3d',
        mode: 'markers',
        x: scatterData.filter(d => d.label === 0).map(d => d.x),
        y: scatterData.filter(d => d.label === 0).map(d => d.y),
        z: scatterData.filter(d => d.label === 0).map(() => 1.1),
        marker: { size: 8, color: '#3b82f6', symbol: 'circle', line: { color: '#ffffff', width: 1.5 } },
        name: 'Class 0 (truth)',
        hovertemplate: `${featureLabels[sel0]}: %{x:.3f}<br>${featureLabels[sel1]}: %{y:.3f}<br>class: 0<extra></extra>`,
      };

      const traceClass1: any = {
        type: 'scatter3d',
        mode: 'markers',
        x: scatterData.filter(d => d.label === 1).map(d => d.x),
        y: scatterData.filter(d => d.label === 1).map(d => d.y),
        z: scatterData.filter(d => d.label === 1).map(() => 1.1),
        marker: { size: 8, color: '#ef4444', symbol: 'diamond', line: { color: '#ffffff', width: 1.5 } },
        name: 'Class 1 (truth)',
        hovertemplate: `${featureLabels[sel0]}: %{x:.3f}<br>${featureLabels[sel1]}: %{y:.3f}<br>class: 1<extra></extra>`,
      };

      const layout: any = {
        title: { text: 'Decision Surface &bull; VQC Raw PauliZ Output', font: { size: 16 } },
        scene: {
          xaxis: { title: featureLabels[sel0] },
          yaxis: { title: featureLabels[sel1] },
          zaxis: { title: 'Raw Prediction', range: [-1.1, 1.1] },
          camera: { eye: { x: 1.5, y: 1.5, z: 0.8 } },
          aspectmode: 'cube',
        },
        margin: { l: 0, r: 0, t: 40, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        hovermode: 'closest',
        legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0.5)', font: { color: '#e2e8f0' } },
      };

      await window.Plotly.newPlot(plotRef.current, [traceSurface, traceClass0, traceClass1], layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Cube3d className="h-4 w-4" />
          3D Decision Boundary
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Surface shows raw PauliZ expectation (range [-1, +1]) — the decision boundary is where it crosses zero.
          Colored markers show true class labels.
          {numFeatures > 2 && ' Unselected features are fixed at their median value.'}
        </p>

        {numFeatures > 2 && (
          <div className="flex gap-4">
            <div className="flex-1 space-y-1">
              <Label className="text-xs">X-axis feature</Label>
              <select
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm"
                value={featIdx0}
                onChange={e => setFeatIdx0(Number(e.target.value))}
              >
                {featureLabels.map((label, i) => (
                  <option key={i} value={i} disabled={i === featIdx1}>{label}</option>
                ))}
              </select>
            </div>
            <div className="flex-1 space-y-1">
              <Label className="text-xs">Y-axis feature</Label>
              <select
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm"
                value={featIdx1}
                onChange={e => setFeatIdx1(Number(e.target.value))}
              >
                {featureLabels.map((label, i) => (
                  <option key={i} value={i} disabled={i === featIdx0}>{label}</option>
                ))}
              </select>
            </div>
          </div>
        )}

        <Button onClick={handleGenerate} disabled={loading || !plotReady} className="gap-2">
          {loading ? 'Generating...' : 'Generate 3D Plot'}
        </Button>

        {error && (
          <div className="flex items-center gap-2 text-sm text-destructive">
            <AlertCircle className="h-4 w-4" />
            {error}
          </div>
        )}
        <div ref={plotRef} className="w-full h-[500px] rounded-lg border" />
      </CardContent>
    </Card>
  );
}

function linspace(start: number, end: number, n: number): number[] {
  const step = (end - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + i * step);
}
