# Sample Datasets

## parity_train.csv
Binary classification for parity function. Label 1 if odd number of 1s, 0 if even.
- **Features**: 4 binary columns (f1-f4)
- **Labels**: 0 or 1
- **Encoding**: Basis encoding (for binary inputs)
- **Circuit Config**: 4 qubits, 2 layers

## parity_test.csv
Test data for parity classification (unseen samples).
- **Features**: 4 binary columns
- **Labels**: 0 or 1

## iris_classification.csv
Simplified Iris dataset (first 2 classes).
- **Features**: 4 real-valued columns (sepal_length, sepal_width, petal_length, petal_width)
- **Labels**: 0 or 1
- **Encoding**: Amplitude encoding (for real-valued inputs)
- **Circuit Config**: 2 qubits, 6 layers

## simple_classification.csv
Simple 2-feature dataset for quick testing.
- **Features**: 2 real-valued columns
- **Labels**: 0 or 1
- **Encoding**: Amplitude encoding
- **Circuit Config**: 2 qubits, 3 layers

## Usage
Upload any of these CSV files through the Data Upload tab in the application.
