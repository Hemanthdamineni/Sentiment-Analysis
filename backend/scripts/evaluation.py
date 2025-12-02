import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from .data_processing import get_class_names

def compute_metrics(predictions, targets):
    class_names = get_class_names()
    cm = confusion_matrix(targets, predictions)
    report = classification_report(targets, predictions, target_names=class_names, output_dict=True, zero_division=0)
    return {'confusion_matrix': cm, 'classification_report': report}

def plot_confusion_matrix(confusion_matrix, class_names, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix'); plt.tight_layout(); return plt.gcf()

def plot_metrics_history(training_history, figsize=(12, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(training_history['train_loss'], label='Training Loss'); ax1.plot(training_history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss'); ax1.legend()
    ax2.plot(training_history['train_acc'], label='Training Accuracy'); ax2.plot(training_history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Training and Validation Accuracy'); ax2.legend()
    plt.tight_layout(); return fig

def analyze_misclassifications(model, data_loader, device, n_examples=5):
    import torch
    model.eval(); misclassified_examples = []; class_names = get_class_names()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); targets = batch['targets'].to(device)
            review_texts = batch['review_text']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            for i, (pred, target) in enumerate(zip(preds, targets)):
                if pred != target:
                    misclassified_examples.append({'text': review_texts[i], 'predicted': class_names[pred.item()], 'actual': class_names[target.item()]})
                    if len(misclassified_examples) >= n_examples: return misclassified_examples
    return misclassified_examples

def save_evaluation_results(metrics, figures_dict, output_dir):
    import os, json
    os.makedirs(output_dir, exist_ok=True)
    def convert_for_json(obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    metrics_json = {k: convert_for_json(v) for k, v in metrics.items()}
    with open(f"{output_dir}/metrics.json", 'w') as f: json.dump(metrics_json, f, indent=4)
    for name, fig in figures_dict.items(): fig.savefig(f"{output_dir}/{name}.png"); plt.close(fig)

def plot_model_comparison_bar(results_dict, metric_key='accuracy', figsize=(12,8)):
    labels = list(results_dict.keys()); values = [results_dict[k][metric_key] for k in labels]
    plt.figure(figsize=figsize); sns.barplot(x=labels, y=values)
    plt.ylabel(metric_key.capitalize()); plt.xticks(rotation=30, ha='right'); plt.title(f'Model Comparison by {metric_key.capitalize()}'); plt.tight_layout(); return plt.gcf()

def plot_top_tfidf_features(vectorizer, clf, class_names=None, top_n=20, figsize=(12,8)):
    try: feature_names = vectorizer.get_feature_names_out()
    except Exception: feature_names = np.array(list(vectorizer.vocabulary_.keys()))
    coefs = clf.coef_; n_classes = coefs.shape[0]
    if class_names is None: class_names = get_class_names()[:n_classes]
    fig, axes = plt.subplots(n_classes, 1, figsize=figsize); axes = [axes] if n_classes == 1 else axes
    for idx in range(n_classes):
        coef = coefs[idx]; top_idx = np.argsort(coef)[-top_n:]
        top_features = feature_names[top_idx]; top_values = coef[top_idx]; order = np.argsort(top_values)
        top_features = top_features[order]; top_values = top_values[order]
        axes[idx].barh(top_features, top_values); axes[idx].set_title(f'Top TF-IDF Features for class: {class_names[idx]}'); axes[idx].set_xlabel('Coefficient weight')
    plt.tight_layout(); return fig

def plot_sentiment_frequency(df, figsize=(8,6), title=None):
    import pandas as pd
    class_names = get_class_names()
    counts = df['sentiment'].value_counts().reindex([0,1,2], fill_value=0)
    labels = [class_names[i] for i in [0,1,2]]
    values = [int(counts[i]) for i in [0,1,2]]
    plt.figure(figsize=figsize)
    sns.barplot(x=labels, y=values)
    plt.ylabel('Count')
    plt.xlabel('Sentiment')
    if title:
        plt.title(title)
    else:
        plt.title('Sentiment Frequency')
    plt.tight_layout(); return plt.gcf()
