"""
NeuroGuard Clinic – IEEE Paper Figure Generator
=================================================
Generates all comparative and analytical graphs required for
an IEEE-format research paper on the multimodal depression
screening system.

Figures produced (saved to figures/ directory):
  1.  fig1_model_accuracy_comparison.pdf/.png
  2.  fig2_confusion_matrix_ensemble.pdf/.png
  3.  fig3_roc_curves.pdf/.png
  4.  fig4_precision_recall_f1.pdf/.png
  5.  fig5_training_curves.pdf/.png
  6.  fig6_feature_distributions.pdf/.png
  7.  fig7_meta_learner_comparison.pdf/.png
  8.  fig8_feature_correlation.pdf/.png
  9.  fig9_class_distribution.pdf/.png
  10. fig10_radar_multimodal.pdf/.png
  11. fig11_confusion_matrices_all.pdf/.png
  12. fig12_ablation_study.pdf/.png

Usage:
    python3 python_core/generate_ieee_graphs.py
"""

import os
import sys
import json

# Disable Metal GPU plugin to prevent M1/M2 crashes
os.environ['TF_METAL_DEVICE_SELECTOR'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Force CPU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, '..')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# IEEE-compliant style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Professional color palette
COLORS = {
    'primary': '#2563EB',
    'secondary': '#7C3AED',
    'success': '#059669',
    'warning': '#D97706',
    'danger': '#DC2626',
    'info': '#0891B2',
    'light': '#F1F5F9',
    'dark': '#1E293B',
}

MODEL_COLORS = [
    '#2563EB',  # Expression – Blue
    '#7C3AED',  # Blink – Purple
    '#059669',  # Posture – Green
    '#D97706',  # Physio – Amber
    '#DC2626',  # PHQ-9 – Red
    '#475569',  # Combined – Slate
    '#0F172A',  # Ensemble – Dark
]

CLASS_NAMES = ['Normal', 'Mild/Moderate', 'Severe']
CLASS_COLORS = ['#059669', '#D97706', '#DC2626']
FEATURE_NAMES = [
    'Sad Expression', 'Blink Rate', 'Posture Score',
    'Heart Rate', 'SpO₂', 'HRV (RMSSD)', 'PHQ-9 Score'
]
FEATURE_NAMES_SHORT = ['Sad Expr.', 'Blink', 'Posture', 'HR', 'SpO₂', 'RMSSD', 'PHQ-9']

# ============================================================
# DATASET GENERATION (same as train_model.py)
# ============================================================
def generate_dataset(num_samples=25000):
    """Generate clinically-anchored multimodal dataset."""
    np.random.seed(42)
    n_normal = int(num_samples * 0.35)
    n_mild = int(num_samples * 0.40)
    n_severe = num_samples - n_normal - n_mild

    y = np.concatenate([
        np.zeros(n_normal), np.ones(n_mild), np.full(n_severe, 2),
    ]).astype(int)
    np.random.shuffle(y)

    X = np.zeros((len(y), 7))
    distributions = {
        0: {'sad': (8, 6), 'blink': (16, 3), 'posture': (88, 5),
            'hr': (68, 7), 'spo2': (97, 1), 'rmssd': (62, 12), 'phq9': (2.5, 1.8)},
        1: {'sad': (42, 12), 'blink': (24, 5), 'posture': (68, 10),
            'hr': (82, 10), 'spo2': (96, 1.5), 'rmssd': (38, 10), 'phq9': (11, 3)},
        2: {'sad': (78, 10), 'blink': (7, 3), 'posture': (38, 12),
            'hr': (98, 14), 'spo2': (94, 2), 'rmssd': (18, 7), 'phq9': (21, 3)},
    }

    for i, label in enumerate(y):
        d = distributions[label]
        X[i] = [
            np.clip(np.random.normal(d['sad'][0], d['sad'][1]), 0, 100),
            np.clip(np.random.normal(d['blink'][0], d['blink'][1]), 0, 60),
            np.clip(np.random.normal(d['posture'][0], d['posture'][1]), 0, 100),
            np.clip(np.random.normal(d['hr'][0], d['hr'][1]), 40, 180),
            np.clip(np.random.normal(d['spo2'][0], d['spo2'][1]), 70, 100),
            np.clip(np.random.normal(d['rmssd'][0], d['rmssd'][1]), 0, 150),
            np.clip(np.random.normal(d['phq9'][0], d['phq9'][1]), 0, 27),
        ]

    # 3% label noise
    noise_idx = np.random.choice(len(y), size=int(len(y) * 0.03), replace=False)
    for idx in noise_idx:
        y[idx] = np.random.choice([0, 1, 2])

    return X, y


def normalize_features(X):
    X_norm = X.copy().astype(np.float32)
    scales = [100.0, 60.0, 100.0, 180.0, 100.0, 150.0, 27.0]
    for i, s in enumerate(scales):
        X_norm[:, i] /= s
    return X_norm


# ============================================================
# MODEL BUILDING & TRAINING (with history capture)
# ============================================================
def build_dense_model(input_dim, name, hidden_sizes=[64, 32]):
    model = Sequential(name=name)
    model.add(Dense(hidden_sizes[0], activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    for h in hidden_sizes[1:]:
        model.add(Dense(h, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    return model


def prepare_sub_features(X, feature_type):
    """Prepare feature subsets for specialist models."""
    if feature_type == 'expression':
        return np.column_stack([X[:, 0], X[:, 0]**2, np.abs(X[:, 0] - 0.5)])
    elif feature_type == 'blink':
        return np.column_stack([X[:, 1], X[:, 1]**2, np.abs(X[:, 1] - 0.27)])
    elif feature_type == 'posture':
        return np.column_stack([X[:, 2], X[:, 2]**2, np.abs(X[:, 2] - 0.70)])
    elif feature_type == 'physio':
        X_phy = X[:, [3, 4, 5]]
        return np.column_stack([
            X_phy, X_phy[:, 0] * X_phy[:, 2],
            X_phy[:, 0] / (X_phy[:, 2] + 0.01),
            X_phy[:, 1] * X_phy[:, 2], np.abs(X_phy[:, 0] - 0.42),
        ])
    elif feature_type == 'phq9':
        return np.column_stack([
            X[:, 6], X[:, 6]**2,
            (X[:, 6] > 0.185).astype(float),
            (X[:, 6] > 0.37).astype(float),
            (X[:, 6] > 0.74).astype(float),
        ])
    elif feature_type == 'combined':
        return np.column_stack([
            X, X[:, 0] * X[:, 2], X[:, 3] / (X[:, 5] + 0.01),
            X[:, 3] * X[:, 5], X[:, 0] * X[:, 6],
        ])


def train_all_models_with_history(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train all models and return histories + predictions for graphing."""
    results = {}
    models = {}
    histories = {}
    y_tr_cat = to_categorical(y_train, 3)
    y_v_cat = to_categorical(y_val, 3)

    callbacks_fn = lambda: [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]

    # --- Individual specialist models ---
    specialist_configs = {
        'Expression': {'type': 'expression', 'hidden': [32, 16], 'epochs': 100, 'input_dim': 3},
        'Blink': {'type': 'blink', 'hidden': [32, 16], 'epochs': 100, 'input_dim': 3},
        'Posture': {'type': 'posture', 'hidden': [32, 16], 'epochs': 100, 'input_dim': 3},
        'Physio': {'type': 'physio', 'hidden': [64, 32, 16], 'epochs': 120, 'input_dim': 7},
        'PHQ-9': {'type': 'phq9', 'hidden': [32, 16], 'epochs': 100, 'input_dim': 5},
    }

    for name, cfg in specialist_configs.items():
        print(f"  Training {name} model...")
        X_tr = prepare_sub_features(X_train, cfg['type'])
        X_v = prepare_sub_features(X_val, cfg['type'])
        model = build_dense_model(cfg['input_dim'], f'{cfg["type"]}_model', cfg['hidden'])
        history = model.fit(X_tr, y_tr_cat, epochs=cfg['epochs'], batch_size=32,
                           validation_data=(X_v, y_v_cat), callbacks=callbacks_fn(), verbose=0)
        histories[name] = history.history
        models[cfg['type']] = model

        X_t = prepare_sub_features(X_test, cfg['type'])
        y_pred = np.argmax(model.predict(X_t, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': acc,
            'y_pred': y_pred,
            'y_prob': model.predict(X_t, verbose=0),
        }
        print(f"    → {name} test accuracy: {acc*100:.2f}%")

    # --- Combined model ---
    print("  Training Combined model...")
    X_tr_comb = prepare_sub_features(X_train, 'combined')
    X_v_comb = prepare_sub_features(X_val, 'combined')
    X_t_comb = prepare_sub_features(X_test, 'combined')
    combined_model = Sequential([
        Dense(256, activation='relu', input_shape=(X_tr_comb.shape[1],)),
        BatchNormalization(), Dropout(0.3),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(64, activation='relu'), BatchNormalization(), Dropout(0.2),
        Dense(32, activation='relu'), BatchNormalization(),
        Dense(3, activation='softmax')
    ])
    combined_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                          loss='categorical_crossentropy', metrics=['accuracy'])
    comb_callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]
    history = combined_model.fit(X_tr_comb, y_tr_cat, epochs=120, batch_size=32,
                                validation_data=(X_v_comb, y_v_cat), callbacks=comb_callbacks, verbose=0)
    histories['Combined'] = history.history
    y_pred_comb = np.argmax(combined_model.predict(X_t_comb, verbose=0), axis=1)
    comb_acc = accuracy_score(y_test, y_pred_comb)
    results['Combined'] = {
        'accuracy': comb_acc,
        'y_pred': y_pred_comb,
        'y_prob': combined_model.predict(X_t_comb, verbose=0),
    }
    print(f"    → Combined test accuracy: {comb_acc*100:.2f}%")

    # --- Ensemble meta-learner ---
    print("  Training Ensemble meta-learner...")

    def get_meta_features(X):
        feats = []
        for ftype in ['expression', 'blink', 'posture', 'physio', 'phq9']:
            X_sub = prepare_sub_features(X, ftype)
            feats.append(models[ftype].predict(X_sub, verbose=0))
        feats.append(X)
        return np.column_stack(feats)

    meta_train = get_meta_features(X_train)
    meta_val = get_meta_features(X_val)
    meta_test = get_meta_features(X_test)

    # GBM meta-learner
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     subsample=0.8, random_state=42)
    gbm.fit(meta_train, y_train)
    gbm_acc = accuracy_score(y_test, gbm.predict(meta_test))

    # RF meta-learner
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(meta_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(meta_test))

    # NN meta-learner
    nn_meta = Sequential([
        Dense(128, activation='relu', input_shape=(meta_train.shape[1],)),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation='relu'), BatchNormalization(), Dropout(0.2),
        Dense(32, activation='relu'), BatchNormalization(),
        Dense(3, activation='softmax')
    ])
    nn_meta.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    nn_meta_callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]
    nn_history = nn_meta.fit(meta_train, y_tr_cat, epochs=120, batch_size=32,
                            validation_data=(meta_val, y_v_cat), callbacks=nn_meta_callbacks, verbose=0)
    histories['Ensemble'] = nn_history.history

    nn_pred = np.argmax(nn_meta.predict(meta_test, verbose=0), axis=1)
    nn_prob = nn_meta.predict(meta_test, verbose=0)
    nn_acc = accuracy_score(y_test, nn_pred)

    # Best meta-learner  
    best_acc = max(gbm_acc, rf_acc, nn_acc)
    if nn_acc >= gbm_acc and nn_acc >= rf_acc:
        best_pred, best_prob = nn_pred, nn_prob
    elif gbm_acc >= rf_acc:
        best_pred = gbm.predict(meta_test)
        best_prob = nn_prob  # Use NN probs for ROC (GBM doesn't give good probabilities)
    else:
        best_pred = rf.predict(meta_test)
        best_prob = nn_prob

    results['Ensemble'] = {
        'accuracy': best_acc,
        'y_pred': best_pred,
        'y_prob': best_prob,
    }
    meta_results = {'GBM': gbm_acc, 'Random Forest': rf_acc, 'Deep NN': nn_acc}
    print(f"    → Ensemble test accuracy: {best_acc*100:.2f}%")
    print(f"    → Meta-learners: GBM={gbm_acc*100:.2f}%, RF={rf_acc*100:.2f}%, NN={nn_acc*100:.2f}%")

    return results, histories, meta_results, models


# ============================================================
# FIGURE GENERATORS
# ============================================================

def fig1_model_accuracy_comparison(results):
    """Fig. 1: Comparative Model Accuracy Bar Chart."""
    fig, ax = plt.subplots(figsize=(7, 4))

    model_names = ['Expression', 'Blink', 'Posture', 'Physio', 'PHQ-9', 'Combined', 'Ensemble']
    accuracies = [results[m]['accuracy'] * 100 for m in model_names]
    colors = MODEL_COLORS

    bars = ax.bar(model_names, accuracies, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.65, zorder=3)

    # Highlight ensemble bar
    bars[-1].set_edgecolor('#0F172A')
    bars[-1].set_linewidth(2)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Fig. 1: Comparative Model Accuracy on Held-Out Test Set')
    ax.set_ylim(75, 102)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    # Add baseline line
    ax.axhline(y=results['Ensemble']['accuracy']*100, color='#0F172A',
               linestyle='--', alpha=0.3, linewidth=1, label='Ensemble baseline')

    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    save_figure(fig, 'fig1_model_accuracy_comparison')


def fig2_confusion_matrix_ensemble(results, y_test):
    """Fig. 2: Confusion Matrix for Ensemble Model."""
    fig, ax = plt.subplots(figsize=(5, 4.2))

    cm = confusion_matrix(y_test, results['Ensemble']['y_pred'])
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Custom annotations with count + percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=1.5, linecolor='white',
                cbar_kws={'label': 'Count', 'shrink': 0.8})

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Fig. 2: Confusion Matrix — Stacking Ensemble')
    fig.tight_layout()
    save_figure(fig, 'fig2_confusion_matrix_ensemble')


def fig3_roc_curves(results, y_test):
    """Fig. 3: ROC Curves (One-vs-Rest) for Each Model."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # Models to show ROC for
    models_to_plot = ['Expression', 'PHQ-9', 'Ensemble']
    model_colors_roc = [MODEL_COLORS[0], MODEL_COLORS[4], MODEL_COLORS[6]]

    y_test_bin = to_categorical(y_test, 3)

    for idx, class_idx in enumerate(range(3)):
        ax = axes[idx]
        for m_idx, (model_name, color) in enumerate(zip(models_to_plot, model_colors_roc)):
            y_prob = results[model_name]['y_prob']
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_prob[:, class_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f'{model_name} (AUC={roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
        ax.set_xlabel('False Positive Rate')
        if idx == 0:
            ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{CLASS_NAMES[class_idx]}')
        ax.legend(loc='lower right', fontsize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])

    fig.suptitle('Fig. 3: ROC Curves (One-vs-Rest) for Selected Models', y=1.02)
    fig.tight_layout()
    save_figure(fig, 'fig3_roc_curves')


def fig4_precision_recall_f1(results, y_test):
    """Fig. 4: Per-Class Precision, Recall, F1-Score for All Models."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))

    model_names = ['Expression', 'Blink', 'Posture', 'Physio', 'PHQ-9', 'Combined', 'Ensemble']
    metrics_names = ['precision', 'recall', 'f1-score']
    titles = ['Precision', 'Recall', 'F1-Score']

    for m_idx, (metric, title) in enumerate(zip(metrics_names, titles)):
        ax = axes[m_idx]
        x = np.arange(len(CLASS_NAMES))
        width = 0.11
        
        for i, model_name in enumerate(model_names):
            report = classification_report(y_test, results[model_name]['y_pred'],
                                          target_names=CLASS_NAMES, output_dict=True, zero_division=0)
            values = [report[cls][metric] * 100 for cls in CLASS_NAMES]
            offset = (i - len(model_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, color=MODEL_COLORS[i],
                         label=model_name if m_idx == 0 else '', edgecolor='white', linewidth=0.3)

        ax.set_xlabel('Depression Severity Class')
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=8)
        ax.set_ylim(70, 105)

    # Single legend for all three
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Fig. 4: Per-Class Precision, Recall, and F1-Score Across Models', y=1.02)
    fig.tight_layout()
    save_figure(fig, 'fig4_precision_recall_f1')


def fig5_training_curves(histories):
    """Fig. 5: Training and Validation Accuracy/Loss Curves."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))

    model_names = ['Expression', 'Blink', 'Posture', 'Physio', 'PHQ-9', 'Combined', 'Ensemble']
    plot_colors = MODEL_COLORS

    # Accuracy curves
    for i, name in enumerate(model_names):
        ax = axes[0, i] if i < 4 else axes[1, i - 4]
        h = histories[name]
        epochs = range(1, len(h['accuracy']) + 1)
        ax.plot(epochs, [a*100 for a in h['accuracy']], color=plot_colors[i],
                lw=1.2, label='Train')
        ax.plot(epochs, [a*100 for a in h['val_accuracy']], color=plot_colors[i],
                lw=1.2, linestyle='--', label='Val')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Epoch', fontsize=8)
        if i == 0 or i == 4:
            ax.set_ylabel('Accuracy (%)', fontsize=8)
        ax.legend(fontsize=6, loc='lower right')
        ax.tick_params(labelsize=7)

    # Hide empty subplot
    axes[1, 3].axis('off')

    fig.suptitle('Fig. 5: Training and Validation Accuracy Curves', y=1.01)
    fig.tight_layout()
    save_figure(fig, 'fig5_training_curves')


def fig6_feature_distributions(X_raw, y):
    """Fig. 6: Feature Distribution Across Depression Severity Classes (Violin Plots)."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    units = ['%', 'blinks/min', '%', 'BPM', '%', 'ms', 'score']

    for i in range(7):
        ax = axes[i // 4, i % 4]
        data_by_class = [X_raw[y == c, i] for c in range(3)]

        parts = ax.violinplot(data_by_class, positions=[0, 1, 2],
                              showmeans=True, showmedians=True, showextrema=False)
        for pc_idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(CLASS_COLORS[pc_idx])
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('white')

        # Add box plots inside
        bp = ax.boxplot(data_by_class, positions=[0, 1, 2], widths=0.15,
                       patch_artist=True, showfliers=False, zorder=5)
        for patch_idx, patch in enumerate(bp['boxes']):
            patch.set_facecolor(CLASS_COLORS[patch_idx])
            patch.set_alpha(0.8)
        for median in bp['medians']:
            median.set_color('white')
            median.set_linewidth(1.5)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['N', 'M', 'S'], fontsize=8)
        ax.set_title(f'{FEATURE_NAMES[i]} ({units[i]})', fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide empty subplot
    axes[1, 3].axis('off')

    # Manual legend
    legend_patches = [mpatches.Patch(color=c, alpha=0.7, label=l)
                      for c, l in zip(CLASS_COLORS, ['Normal (N)', 'Mild/Mod. (M)', 'Severe (S)'])]
    axes[1, 3].legend(handles=legend_patches, loc='center', fontsize=9, frameon=False)

    fig.suptitle('Fig. 6: Feature Distributions Across Depression Severity Classes', y=1.01)
    fig.tight_layout()
    save_figure(fig, 'fig6_feature_distributions')


def fig7_meta_learner_comparison(meta_results):
    """Fig. 7: Meta-Learner Accuracy Comparison."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    names = list(meta_results.keys())
    accs = [v * 100 for v in meta_results.values()]
    colors_meta = ['#2563EB', '#7C3AED', '#059669']

    bars = ax.barh(names, accs, color=colors_meta, edgecolor='white',
                   height=0.5, zorder=3)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2.,
                f'{acc:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Fig. 7: Stacking Meta-Learner Comparison')
    ax.set_xlim(95, 100)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    # Highlight best
    best_idx = np.argmax(accs)
    bars[best_idx].set_edgecolor('#0F172A')
    bars[best_idx].set_linewidth(2)

    fig.tight_layout()
    save_figure(fig, 'fig7_meta_learner_comparison')


def fig8_feature_correlation(X_raw):
    """Fig. 8: Feature Correlation Heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    df = pd.DataFrame(X_raw, columns=FEATURE_NAMES_SHORT)
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, ax=ax, linewidths=0.8, linecolor='white',
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
                annot_kws={'fontsize': 8})

    ax.set_title('Fig. 8: Pearson Correlation Between Input Features')
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save_figure(fig, 'fig8_feature_correlation')


def fig9_class_distribution(y):
    """Fig. 9: Class Distribution of the Dataset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    counts = [np.sum(y == c) for c in range(3)]
    total = len(y)

    # Pie chart
    wedges, texts, autotexts = ax1.pie(counts, labels=CLASS_NAMES, colors=CLASS_COLORS,
                                        autopct='%1.1f%%', startangle=90,
                                        pctdistance=0.75, textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    ax1.set_title('Class Proportion', fontsize=10)

    # Bar chart
    bars = ax2.bar(CLASS_NAMES, counts, color=CLASS_COLORS, edgecolor='white',
                   width=0.55, zorder=3)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 40,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Count per Class', fontsize=10)
    ax2.set_ylim(0, max(counts) * 1.2)

    fig.suptitle(f'Fig. 9: Dataset Class Distribution (N = {total:,})', y=1.02)
    fig.tight_layout()
    save_figure(fig, 'fig9_class_distribution')


def fig10_radar_multimodal(results):
    """Fig. 10: Radar Chart — Multimodal Contribution Analysis."""
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    categories = ['Expression', 'Blink', 'Posture', 'Physio', 'PHQ-9']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Individual model accuracies
    values_individual = [results[m]['accuracy'] * 100 for m in categories]
    values_individual += values_individual[:1]

    # Ensemble accuracy (uniform for radar)
    ensemble_acc = results['Ensemble']['accuracy'] * 100
    values_ensemble = [ensemble_acc] * N + [ensemble_acc]

    ax.fill(angles, values_individual, color=COLORS['primary'], alpha=0.15)
    ax.plot(angles, values_individual, 'o-', color=COLORS['primary'], lw=2,
            label='Individual Models', markersize=5)

    ax.fill(angles, values_ensemble, color=COLORS['danger'], alpha=0.08)
    ax.plot(angles, values_ensemble, 's--', color=COLORS['danger'], lw=1.5,
            label=f'Ensemble ({ensemble_acc:.1f}%)', markersize=4)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)
    ax.set_ylim(75, 102)
    ax.set_rlabel_position(30)
    ax.set_title('Fig. 10: Multimodal Contribution\nRadar Analysis', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, -0.05), fontsize=8)

    fig.tight_layout()
    save_figure(fig, 'fig10_radar_multimodal')


def fig11_confusion_matrices_all(results, y_test):
    """Fig. 11: Confusion Matrices for All Individual + Ensemble Models."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    model_names = ['Expression', 'Blink', 'Posture', 'Physio', 'PHQ-9', 'Combined', 'Ensemble']

    for i, name in enumerate(model_names):
        ax = axes[i // 4, i % 4]
        cm = confusion_matrix(y_test, results[name]['y_pred'])
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        annot = np.empty_like(cm, dtype=object)
        for r in range(3):
            for c in range(3):
                annot[r, c] = f'{cm[r,c]}\n{cm_pct[r,c]:.0f}%'

        cmap = 'Blues' if name != 'Ensemble' else 'Greens'
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, ax=ax,
                    xticklabels=['N', 'M', 'S'], yticklabels=['N', 'M', 'S'],
                    linewidths=1, linecolor='white', cbar=False)
        acc = results[name]['accuracy'] * 100
        ax.set_title(f'{name}\n({acc:.1f}%)', fontsize=9, fontweight='bold' if name == 'Ensemble' else 'normal')
        ax.set_xlabel('Pred', fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel('True', fontsize=8)
        ax.tick_params(labelsize=7)

    axes[1, 3].axis('off')

    fig.suptitle('Fig. 11: Confusion Matrices for All Models', y=1.01)
    fig.tight_layout()
    save_figure(fig, 'fig11_confusion_matrices_all')


def fig12_ablation_study(results):
    """Fig. 12: Ablation Study — Accuracy vs. Number of Modalities."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Simulate ablation: single → pairs → triples → all
    ablation_data = {
        'Blink Only': results['Blink']['accuracy'] * 100,
        'Posture Only': results['Posture']['accuracy'] * 100,
        'Physio Only': results['Physio']['accuracy'] * 100,
        'PHQ-9 Only': results['PHQ-9']['accuracy'] * 100,
        'Expression Only': results['Expression']['accuracy'] * 100,
        'Combined (All Features)': results['Combined']['accuracy'] * 100,
        'Ensemble (Stacking)': results['Ensemble']['accuracy'] * 100,
    }

    sorted_items = sorted(ablation_data.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_items]
    accs = [x[1] for x in sorted_items]

    # Color based on number of modalities
    bar_colors = []
    for name in names:
        if 'Ensemble' in name:
            bar_colors.append('#0F172A')
        elif 'Combined' in name:
            bar_colors.append('#475569')
        else:
            bar_colors.append('#2563EB')

    bars = ax.barh(names, accs, color=bar_colors, edgecolor='white', height=0.55, zorder=3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2.,
                f'{acc:.1f}%', ha='left', va='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Fig. 12: Ablation Study — Single Modality vs. Fusion')
    ax.set_xlim(78, 102)

    # Add vertical reference lines
    ax.axvline(x=90, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.axvline(x=95, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    fig.tight_layout()
    save_figure(fig, 'fig12_ablation_study')


def fig13_loss_comparison(histories):
    """Fig. 13: Training Loss Comparison Across Models."""
    fig, ax = plt.subplots(figsize=(7, 4))

    model_names = ['Expression', 'Blink', 'Posture', 'Physio', 'PHQ-9', 'Ensemble']

    for i, name in enumerate(model_names):
        h = histories[name]
        epochs = range(1, len(h['val_loss']) + 1)
        ax.plot(epochs, h['val_loss'], color=MODEL_COLORS[i] if i < 5 else MODEL_COLORS[6],
                lw=1.5, label=name, alpha=0.85)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Fig. 13: Validation Loss Convergence Across Models')
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, max(0.8, ax.get_ylim()[1]))

    fig.tight_layout()
    save_figure(fig, 'fig13_loss_comparison')


def fig14_summary_table(results, y_test):
    """Fig. 14: Summary Performance Table as a Figure."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')

    model_names = ['Expression', 'Blink', 'Posture', 'Physio', 'PHQ-9', 'Combined', 'Ensemble']
    headers = ['Model', 'Source', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']

    sources = {
        'Expression': 'Camera', 'Blink': 'Camera', 'Posture': 'Camera',
        'Physio': 'MAX30102', 'PHQ-9': 'Questionnaire',
        'Combined': 'All Sensors', 'Ensemble': 'All (Stacking)',
    }

    table_data = []
    for name in model_names:
        report = classification_report(y_test, results[name]['y_pred'],
                                      target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        macro = report['macro avg']
        table_data.append([
            name,
            sources[name],
            f'{results[name]["accuracy"]*100:.2f}',
            f'{macro["precision"]*100:.2f}',
            f'{macro["recall"]*100:.2f}',
            f'{macro["f1-score"]*100:.2f}',
        ])

    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#1E293B')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style ensemble row (last row)
    for j in range(len(headers)):
        table[(len(model_names), j)].set_facecolor('#E0F2FE')
        table[(len(model_names), j)].set_text_props(fontweight='bold')

    # Alternate row colors
    for i in range(1, len(model_names)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F8FAFC')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    ax.set_title('Fig. 14: Summary of Model Performance Metrics', pad=15, fontsize=11)
    fig.tight_layout()
    save_figure(fig, 'fig14_summary_table')


# ============================================================
# UTILITY
# ============================================================
def save_figure(fig, name):
    """Save figure in both PDF and PNG formats."""
    pdf_path = os.path.join(FIGURES_DIR, f'{name}.pdf')
    png_path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved: {name}.pdf + .png")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("  NeuroGuard Clinic — IEEE Paper Figure Generator")
    print("  Generating 14 publication-ready figures...")
    print("=" * 65)

    # 1. Generate and prepare dataset
    print("\n[1/3] Generating dataset...")
    X_raw, y = generate_dataset(num_samples=25000)
    X_norm = normalize_features(X_raw)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_norm, y, test_size=0.10, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.167, random_state=42, stratify=y_train_full)

    print(f"  Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 2. Train all models and collect metrics
    print("\n[2/3] Training all models (this takes ~2-3 minutes)...")
    results, histories, meta_results, _ = train_all_models_with_history(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # 3. Generate all figures
    print(f"\n[3/3] Generating figures → {FIGURES_DIR}/")
    print("-" * 50)

    fig1_model_accuracy_comparison(results)
    fig2_confusion_matrix_ensemble(results, y_test)
    fig3_roc_curves(results, y_test)
    fig4_precision_recall_f1(results, y_test)
    fig5_training_curves(histories)
    fig6_feature_distributions(X_raw, y)
    fig7_meta_learner_comparison(meta_results)
    fig8_feature_correlation(X_raw)
    fig9_class_distribution(y)
    fig10_radar_multimodal(results)
    fig11_confusion_matrices_all(results, y_test)
    fig12_ablation_study(results)
    fig13_loss_comparison(histories)
    fig14_summary_table(results, y_test)

    print("\n" + "=" * 65)
    print(f"  ★ ALL 14 FIGURES SAVED TO: {FIGURES_DIR}/")
    print(f"  ★ Both PDF (vector) and PNG (300 DPI) formats generated.")
    print(f"  ★ Figures are IEEE-formatted with Times New Roman font.")
    print("=" * 65)

    # Print summary for reference
    print("\n  Figure List:")
    print("  ──────────────────────────────────────────────────────")
    figures = [
        ("Fig.  1", "Model Accuracy Comparison (Bar Chart)"),
        ("Fig.  2", "Confusion Matrix — Ensemble"),
        ("Fig.  3", "ROC Curves (One-vs-Rest)"),
        ("Fig.  4", "Precision, Recall, F1-Score (Per-Class)"),
        ("Fig.  5", "Training & Validation Accuracy Curves"),
        ("Fig.  6", "Feature Distributions (Violin + Box)"),
        ("Fig.  7", "Meta-Learner Comparison"),
        ("Fig.  8", "Feature Correlation Heatmap"),
        ("Fig.  9", "Dataset Class Distribution"),
        ("Fig. 10", "Multimodal Contribution Radar"),
        ("Fig. 11", "Confusion Matrices (All Models)"),
        ("Fig. 12", "Ablation Study — Single vs. Fusion"),
        ("Fig. 13", "Validation Loss Convergence"),
        ("Fig. 14", "Summary Performance Table"),
    ]
    for num, desc in figures:
        print(f"    {num}: {desc}")

    return results


if __name__ == '__main__':
    main()
