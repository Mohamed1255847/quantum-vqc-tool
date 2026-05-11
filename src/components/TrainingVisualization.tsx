import { useMemo } from 'react';
import { BarChart3, TrendingDown, Target, Clock } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import type { TrainingHistoryItem } from '@/types';

interface TrainingVisualizationProps {
  data: TrainingHistoryItem[];
}

export function TrainingVisualization({ data }: TrainingVisualizationProps) {
  const latestData = data[data.length - 1];
  const initialData = data[0];
  
  const progressPercent = data.length > 0 ? Math.min(100, (data.length / 100) * 100) : 0;
  const averageTimePerIteration = useMemo(() => {
    if (data.length < 2) return 0;
    return data.length / (data.length);
  }, [data.length]);

  const chartHeight = 200;
  const padding = 40;
  const chartWidth = 600;

  const { costPath, accuracyPath, cePath, costPoints, accuracyPoints, cePoints } = useMemo(() => {
    if (data.length === 0) return { costPath: '', accuracyPath: '', cePath: '', costPoints: [], accuracyPoints: [], cePoints: [] };

    const maxCost = Math.max(1, ...data.map(d => d.cost));
    const maxAccuracy = 1;
    const width = chartWidth - 2 * padding;
    const height = chartHeight - 2 * padding;

    const costPointsArr = data.map((d, i) => ({
      x: padding + (i / Math.max(1, data.length - 1)) * width,
      y: chartHeight - padding - (d.cost / maxCost) * height
    }));

    const accuracyPointsArr = data.map((d, i) => ({
      x: padding + (i / Math.max(1, data.length - 1)) * width,
      y: chartHeight - padding - (d.accuracy / maxAccuracy) * height
    }));

    const hasCeLoss = data.some(d => d.cost_cross_entropy !== undefined);
    const cePointsArr = hasCeLoss ? data.map((d, i) => ({
      x: padding + (i / Math.max(1, data.length - 1)) * width,
      y: chartHeight - padding - ((d.cost_cross_entropy || 0) / maxCost) * height
    })) : [];

    return {
      costPoints: costPointsArr,
      accuracyPoints: accuracyPointsArr,
      cePoints: cePointsArr,
      costPath: costPointsArr.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' '),
      accuracyPath: accuracyPointsArr.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' '),
      cePath: cePointsArr.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
    };
  }, [data]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <Card className="bg-muted/50">
          <CardContent className="pt-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <TrendingDown className="h-4 w-4" />
              Final Cost
            </div>
            <div className="text-2xl font-bold mt-1">
              {latestData?.cost.toFixed(6) || '0'}
            </div>
            <div className="text-xs text-muted-foreground">
              Started: {initialData?.cost.toFixed(4) || '0'}
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
              {((latestData?.accuracy || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground">
              Started: {((initialData?.accuracy || 0) * 100).toFixed(1)}%
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
              {data.length}
            </div>
            <Progress value={progressPercent} className="mt-2 h-2" />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <BarChart3 className="h-4 w-4" />
            Training Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative overflow-hidden">
            <svg
              viewBox={`0 0 ${chartWidth} ${chartHeight}`}
              className="w-full h-auto"
              style={{ maxHeight: '300px' }}
            >
              <rect x={padding} y={padding} width={chartWidth - 2 * padding} height={chartHeight - 2 * padding} className="fill-muted/30" />
              
              {data.length > 0 && (
                <>
                  <path d={costPath} className="fill-none stroke-red-500 stroke-2" />
                  {cePath && <path d={cePath} className="fill-none stroke-orange-500 stroke-2" />}
                  <path d={accuracyPath} className="fill-none stroke-green-500 stroke-2" />
                  <circle cx={costPoints[costPoints.length - 1]?.x} cy={costPoints[costPoints.length - 1]?.y} r={4} className="fill-red-500" />
                  {cePoints.length > 0 && <circle cx={cePoints[cePoints.length - 1]?.x} cy={cePoints[cePoints.length - 1]?.y} r={4} className="fill-orange-500" />}
                  <circle cx={accuracyPoints[accuracyPoints.length - 1]?.x} cy={accuracyPoints[accuracyPoints.length - 1]?.y} r={4} className="fill-green-500" />
                </>
              )}
              
              <g className="text-xs fill-muted-foreground">
                <text x={chartWidth / 2} y={chartHeight - 5} textAnchor="middle">Iteration</text>
                <text x={15} y={chartHeight / 2} textAnchor="middle" transform={`rotate(-90, 15, ${chartHeight / 2})`}>Value</text>
              </g>
            </svg>

            <div className="flex justify-center gap-6 mt-2">
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-red-500" />
                <span className="text-sm">Square Loss</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-orange-500" />
                <span className="text-sm">Cross-Entropy</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-green-500" />
                <span className="text-sm">Accuracy</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Training History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-h-[200px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-background">
                <tr className="border-b">
                  <th className="px-2 py-1 text-left">Iter</th>
                  <th className="px-2 py-1 text-right">Square Loss</th>
                  {data.some(d => d.cost_cross_entropy !== undefined) && (
                    <th className="px-2 py-1 text-right">Cross-Ent</th>
                  )}
                  <th className="px-2 py-1 text-right">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {data.slice(-20).map((item) => (
                  <tr key={item.iteration} className="border-b hover:bg-muted/50">
                    <td className="px-2 py-1">{item.iteration}</td>
                    <td className="px-2 py-1 text-right">{item.cost.toFixed(6)}</td>
                    {item.cost_cross_entropy !== undefined && (
                      <td className="px-2 py-1 text-right">{item.cost_cross_entropy.toFixed(6)}</td>
                    )}
                    <td className="px-2 py-1 text-right">{(item.accuracy * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}