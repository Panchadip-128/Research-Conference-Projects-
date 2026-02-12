# SDN IEEE CONECCT Project

This repository contains the refactored code for the Drift-Aware GATv2 model for IoT security.

## Project Structure

*   `src/`: Contains all source code modules.
    *   `config.py`: Configuration parameters.
    *   `data_loader.py`: Data loading and scanning logic.
    *   `graph_builder.py`: Graph construction logic.
    *   `model.py`: GATv2 model architecture.
    *   `train.py`: Training loop and drift detection.
    *   `visualization.py`: Plotting utilities.
*   `notebooks/`: Contains the original Kaggle notebook.
*   `docs/`: Documentation files.
*   `checkpoints/`: Model checkpoints (ignored by git).

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Ensure your data is at the path specified in `src/config.py` (or update `RAW_DATA_PATH`).

## Usage

You can import modules in `src` to run training or inference.
Example:
```python
from src.data_loader import load_and_preprocess_data
from src.graph_builder import create_graph_dataset, get_data_loaders
from src.model import GATv2Classifier
from src.train import train_model, evaluate_model
from src.visualization import plot_confusion_matrix, plot_efficiency

# 1. Load Data
df = load_and_preprocess_data()

# 2. Build Graphs
if df is not None:
    graphs = create_graph_dataset(df)
    train_loader, test_loader = get_data_loaders(graphs)

    # 3. Init Model
    model = GATv2Classifier(num_features=2, hidden_channels=16, num_classes=1)

    # 4. Train
    model, comms = train_model(model, train_loader)

    # 5. Evaluate
    y_true, y_pred = evaluate_model(model, test_loader)
    
    # 6. Visualize
    plot_confusion_matrix(y_true, y_pred)
    plot_efficiency(comms)
```
