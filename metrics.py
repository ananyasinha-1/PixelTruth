"""
Model Analytics Module - Deepfake Detection Metrics & Visualizations
This module provides helper functions to display model performance metrics,
confusion matrices, ROC curves, and dataset distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import io


# ==================== SAMPLE METRICS ====================
def get_sample_metrics():
    """
    Returns sample model performance metrics.
    These are realistic values based on typical deepfake detection models.
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1_score
    """
    return {
        "accuracy": 95.2,
        "precision": 94.8,
        "recall": 95.7,
        "f1_score": 95.2,
    }


# ==================== CONFUSION MATRIX ====================
def get_confusion_matrix_plot():
    """
    Generates a confusion matrix heatmap for binary classification
    (Real vs Fake deepfake detection).
    
    Returns:
        matplotlib figure object
    """
    # Sample confusion matrix: [True Negatives, False Positives]
    #                          [False Negatives, True Positives]
    # For a model with 95% accuracy on ~1000 test samples
    cm = np.array([[475, 25],    # Real images: 475 correct, 25 misclassified
                   [23, 477]])   # Fake images: 23 misclassified, 477 correct
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='RdYlGn',
        cbar=True,
        ax=ax,
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_ylabel('True Label', color='#e5e7eb')
    ax.set_xlabel('Predicted Label', color='#e5e7eb')
    ax.set_title('Confusion Matrix - Deepfake Detection', color='#e5e7eb', fontsize=14, pad=20)
    
    # Style adjustments for dark theme
    ax.tick_params(colors='#e5e7eb')
    for spine in ax.spines.values():
        spine.set_color('#475569')
    
    plt.tight_layout()
    return fig


# ==================== ROC CURVE ====================
def get_roc_curve_plot():
    """
    Generates a ROC (Receiver Operating Characteristic) curve.
    Shows the trade-off between True Positive Rate and False Positive Rate.
    
    Returns:
        matplotlib figure object
    """
    # Sample data: true labels (0=Real, 1=Fake) and predicted probabilities
    y_true = np.array([0]*500 + [1]*500)  # 500 real, 500 fake
    
    # Predicted probabilities (higher = more confident it's fake)
    # Model performs well, so real images get low scores, fake get high scores
    y_prob = np.concatenate([
        np.random.beta(2, 5, 500),      # Real images: mostly low scores
        np.random.beta(5, 2, 500)       # Fake images: mostly high scores
    ])
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='#22c55e', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color='#e5e7eb', fontsize=12)
    ax.set_ylabel('True Positive Rate', color='#e5e7eb', fontsize=12)
    ax.set_title('ROC Curve - Model Discriminative Ability', color='#e5e7eb', fontsize=14, pad=20)
    ax.legend(loc="lower right", facecolor='#1e293b', edgecolor='#475569', labelcolor='#e5e7eb')
    ax.grid(True, alpha=0.2, color='#475569')
    
    # Style adjustments
    ax.tick_params(colors='#e5e7eb')
    for spine in ax.spines.values():
        spine.set_color('#475569')
    
    plt.tight_layout()
    return fig


# ==================== DATASET DISTRIBUTION ====================
def get_dataset_distribution_plot():
    """
    Generates a class distribution chart showing the breakdown of
    Real vs Fake images in the training/testing dataset.
    
    Returns:
        matplotlib figure object
    """
    # Sample dataset distribution
    labels = ['Real Images', 'Fake Images']
    sizes = [5000, 5000]  # Balanced dataset
    colors = ['#22c55e', '#ef4444']
    explode = (0.05, 0.05)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
        textprops={'color': '#e5e7eb', 'fontsize': 11}
    )
    
    # Style percentage text
    for autotext in autotexts:
        autotext.set_color('#0f172a')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Add legend with counts
    ax.legend(
        [f'{labels[0]} ({sizes[0]:,})', f'{labels[1]} ({sizes[1]:,})'],
        loc='upper left',
        bbox_to_anchor=(0.85, 1),
        facecolor='#1e293b',
        edgecolor='#475569',
        labelcolor='#e5e7eb'
    )
    
    ax.set_title('Dataset Distribution', color='#e5e7eb', fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig


# ==================== CLASS STATISTICS ====================
def get_class_statistics():
    """
    Returns statistics for each class (Real and Fake).
    
    Returns:
        dict: Statistics including samples, accuracy per class, etc.
    """
    return {
        "Real": {
            "total_samples": 5000,
            "correctly_classified": 4750,
            "misclassified": 250,
            "class_accuracy": 95.0,
        },
        "Fake": {
            "total_samples": 5000,
            "correctly_classified": 4760,
            "misclassified": 240,
            "class_accuracy": 95.2,
        },
    }
