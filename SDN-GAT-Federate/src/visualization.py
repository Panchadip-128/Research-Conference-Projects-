import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix_final.png'):
    """
    Plots and saves the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Final Confusion Matrix (Weighted GATv2)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(filename)
    # plt.show() # Commented out to avoid blocking execution in some environments
    print(f"Confusion matrix saved to {filename}")

def plot_efficiency(communications, num_rounds=30, filename='efficiency_plot_final.png'):
    """
    Plots and saves the communication efficiency chart.
    """
    plt.figure(figsize=(10, 6))
    baseline = np.cumsum([1] * num_rounds)
    ours = np.cumsum(communications)
    plt.plot(range(1, num_rounds + 1), baseline, '--', label='Standard FedAvg', color='grey')
    plt.plot(range(1, num_rounds + 1), ours, '-o', label='Drift-GATv2 (Ours)', color='green', linewidth=2)
    
    # Highlight the Gap (The "Savings")
    # Note: Using logic from the "Money Shot" cell
    final_savings = baseline[-1] - ours[-1]
    pct_saved = (final_savings / baseline[-1]) * 100

    plt.annotate(f'Bandwidth Saved: {pct_saved:.1f}%', 
             xy=(num_rounds, ours[-1]), 
             xytext=(num_rounds - 10, 20),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='green', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=2))

    plt.xlabel('Round')
    plt.ylabel('Cumulative Cost')
    plt.title('Communication Efficiency')
    plt.legend()
    plt.savefig(filename)
    # plt.show()
    print(f"Efficiency plot saved to {filename}")
    
    return pct_saved
