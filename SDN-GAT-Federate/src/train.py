import torch
from scipy.stats import ks_2samp
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .config import DRIFT_THRESHOLD, POS_WEIGHT, LEARN_RATE, WEIGHT_DECAY, NUM_ROUNDS # Note: Check import names against config.py

def train_model(model, train_loader, num_rounds=NUM_ROUNDS, drift_threshold=DRIFT_THRESHOLD, learning_rate=0.005, weight_decay=1e-4, pos_weight=POS_WEIGHT):
    """
    Implements the Drift-Aware training loop.
    """
    print("\n🚀 Starting Drift-Aware Training...")
    
    # Weighted Loss (Fix for Recall) -> Moved from global scope
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history_loss = []
    communications = []

    model.train()
    for round_num in range(1, num_rounds + 1):
        current_losses = []
        with torch.no_grad():
            for data in train_loader:
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.view(-1, 1))
                current_losses.append(loss.item())
                
        if len(history_loss) > 0:
            ks_stat, _ = ks_2samp(history_loss, current_losses)
        else:
            ks_stat = 1.0

        history_loss = current_losses

        if ks_stat < drift_threshold and round_num > 5:
            print(f"Round {round_num}: 🟢 Low Drift ({ks_stat:.4f}). SKIPPED.")
            communications.append(0)
        else:
            print(f"Round {round_num}: ⚠️ High Drift ({ks_stat:.4f}). UPDATING.")
            communications.append(1)
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.view(-1, 1))
                loss.backward()
                optimizer.step()
                
    return model, communications

def evaluate_model(model, test_loader):
    """
    Evaluates the model and prints classification report.
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = (torch.sigmoid(out) > 0.5).float()
            y_true.extend(data.y.numpy())
            y_pred.extend(pred.numpy())
            
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack']))
    
    return y_true, y_pred
