import { Settings, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import type { CircuitConfig as CircuitConfigType } from '@/types';

interface CircuitConfigProps {
  config: CircuitConfigType;
  onChange: (config: CircuitConfigType) => void;
}

export function CircuitConfig({ config, onChange }: CircuitConfigProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Quantum Circuit Configuration
        </CardTitle>
        <CardDescription>
          Configure the variational quantum circuit parameters
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Number of Qubits: {config.num_qubits}</Label>
              <Slider
                value={[config.num_qubits]}
                min={2}
                max={8}
                step={1}
                onValueChange={([value]) => onChange({ ...config, num_qubits: value })}
              />
              <p className="text-sm text-muted-foreground">
                More qubits can capture more complex patterns but require more computational resources.
              </p>
            </div>

            <div className="space-y-2">
              <Label>Number of Layers: {config.num_layers}</Label>
              <Slider
                value={[config.num_layers]}
                min={1}
                max={10}
                step={1}
                onValueChange={([value]) => onChange({ ...config, num_layers: value })}
              />
              <p className="text-sm text-muted-foreground">
                More layers increase expressibility but may lead to overfitting.
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Encoding Type</Label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant={config.encoding === 'basis' ? 'default' : 'outline'}
                  onClick={() => onChange({ ...config, encoding: 'basis' })}
                  className="h-auto py-2"
                >
                  <div className="text-left">
                    <div className="font-medium">Basis Encoding</div>
                    <div className="text-xs opacity-80">Binary strings</div>
                  </div>
                </Button>
                <Button
                  variant={config.encoding === 'amplitude' ? 'default' : 'outline'}
                  onClick={() => onChange({ ...config, encoding: 'amplitude' })}
                  className="h-auto py-2"
                >
                  <div className="text-left">
                    <div className="font-medium">Amplitude Encoding</div>
                    <div className="text-xs opacity-80">Real vectors</div>
                  </div>
                </Button>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-muted">
              <div className="flex items-start gap-2">
                <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <h4 className="font-medium mb-1">
                    {config.encoding === 'basis' ? 'Basis Encoding' : 'Amplitude Encoding'}
                  </h4>
                  <p className="text-muted-foreground">
                    {config.encoding === 'basis'
                      ? 'Encodes binary inputs (0s and 1s) directly as computational basis states. Best for parity and similar binary classification tasks.'
                      : 'Encodes real-valued vectors as quantum state amplitudes. Best for datasets like Iris with continuous features.'
                    }
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="border-t pt-4">
          <h3 className="font-medium mb-2">Circuit Structure</h3>
          <div className="p-4 rounded-lg bg-muted font-mono text-sm">
            <pre className="whitespace-pre-wrap">
{`Circuit: variational_layer × ${config.num_layers}
  └─ Layer: Rotations(${config.num_qubits} qubits) + CNOT entangling

Total parameters: ${config.num_layers * config.num_qubits * 3} rotation angles
  └─ Each layer: ${config.num_qubits} qubits × 3 rotations (Rx, Ry, Rz)`}
            </pre>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
