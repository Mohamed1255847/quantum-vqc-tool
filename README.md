# Quantum Variational Classifier (VQC) Tool

A web-based tool for researchers to train and use variational quantum classifiers, inspired by [PennyLane's Variational Classifier Tutorial](https://pennylane.ai/qml/demos/tutorial_variational_classifier).

## Features

- **Data Upload**: Upload CSV datasets or use built-in sample data (Parity, Iris)
- **Circuit Configuration**: Visual configuration of qubits, layers, and encoding types
- **Training**: Train quantum circuits with configurable optimizers and parameters
- **Visualization**: Real-time charts showing cost and accuracy during training
- **Model Persistence**: Save and load trained models
- **Both Encoding Types**: 
  - Basis encoding (for binary strings/parity)
  - Amplitude encoding (for real-valued vectors)

## Tech Stack

- **Frontend**: React + Vite + TypeScript + Tailwind CSS + shadcn/ui
- **Backend**: Python FastAPI + PennyLane
- **Quantum**: PennyLane with default.qubit simulator

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Start Backend

```bash
cd backend
python app.py
# Runs on http://localhost:8000
```

### Frontend

```bash
cd quantum-vqc-tool
npm install
npm run dev
# Runs on http://localhost:5173
```

## Usage

1. **Start Backend**: `cd backend && python app.py`
2. **Start Frontend**: `cd quantum-vqc-tool && npm run dev`
3. Open http://localhost:5173 in your browser
4. Upload or load sample data
5. Configure circuit parameters
6. Train and visualize results
7. Save your trained model

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API health check |
| POST | `/train` | Train a new model |
| POST | `/predict` | Make predictions |
| GET | `/models` | List saved models |
| GET | `/models/{id}` | Get specific model |
| POST | `/models/save/{id}` | Save a model |
| GET | `/preprocessing-info` | Get encoding info |
| GET | `/default-configs` | Get default configurations |

## Project Structure

```
quantum-vqc-tool/
├── src/
│   ├── components/
│   │   ├── ui/           # shadcn/ui components
│   │   ├── DataUpload.tsx
│   │   ├── CircuitConfig.tsx
│   │   ├── TrainingVisualization.tsx
│   │   └── ModelManager.tsx
│   ├── hooks/
│   │   └── useTraining.ts
│   ├── lib/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   └── index.css
├── backend/
│   ├── app.py            # FastAPI application
│   ├── models.py         # Pydantic models
│   └── vqc.py            # VQC implementation
└── package.json
```

## Encoding Types

### Basis Encoding
- Encodes binary inputs (0s and 1s) as computational basis states
- Example: `[0, 1, 0, 1]` → quantum state `|0101⟩`
- Best for: Parity function, binary classification

### Amplitude Encoding
- Encodes real-valued vectors as quantum state amplitudes
- Preprocessing: padding → normalization → angle extraction
- Best for: Iris dataset, continuous features
