import os
import torch
from torch.optim import AdamW
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from scripts.data_processing import load_data, create_data_loaders
from scripts.transformers import SentimentClassifier, train_epoch, eval_model, save_model
from scripts.evaluation import plot_metrics_history, plot_sentiment_frequency
import torch.nn as nn
from scripts.classical_ml import train_all_classical_models
import json

def run_train(use_huggingface=True, sample_size=100, output_dir="backend/models/sentiment", epochs=1, batch_size=16, max_length=128, learning_rate=2e-5, weight_decay=0.0, use_class_weights=False, file_path=None, warmup_ratio=0.1, dropout=0.4, gradient_accumulation_steps=1, use_weighted_sampler=False, eval_interval=None, eval_max_batches=1):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    try:
        import torch_directml
        _dml = torch_directml.device()
    except Exception:
        _dml = None
    device = torch.device('cuda') if torch.cuda.is_available() else (_dml if _dml is not None else torch.device('cpu'))
    print(f"Using device: {device}")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_df, test_df = load_data(file_path=file_path, use_huggingface=use_huggingface, sample_size=sample_size)
    train_loader, val_loader, _ = create_data_loaders(train_df, test_df, tokenizer, batch_size=batch_size, max_length=max_length, use_weighted_sampler=use_weighted_sampler)
    train_freq_fig = plot_sentiment_frequency(train_df, title='Sentiment Frequency – Train')
    train_freq_fig.savefig(os.path.join(output_dir, "results", "sentiment_frequency_train.png"))
    test_freq_fig = plot_sentiment_frequency(test_df, title='Sentiment Frequency – Test')
    test_freq_fig.savefig(os.path.join(output_dir, "results", "sentiment_frequency_test.png"))
    model = SentimentClassifier(n_classes=3, dropout_p=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * max(1, epochs) // max(1, gradient_accumulation_steps)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    best_accuracy = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    print("\nData preprocessing and loader setup complete.")
    # print(f"Train size: {len(train_df)}, Test size: {len(test_df)}, Batch size: {batch_size}, Max length: {max_length}")
    # optional class-weighted loss for imbalance
    loss_fn = None
    if use_class_weights:
        counts = train_df['sentiment'].value_counts().to_dict()
        total = sum(counts.values())
        weights = [total / (counts.get(i, 1) * len(counts)) for i in range(3)]
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        # print(f"Using class weights: {weights}")

    print("="*50, "\nTRAINING TRANSFORMER\n", "="*50)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}:")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler, loss_fn=loss_fn, gradient_accumulation_steps=gradient_accumulation_steps, eval_loader=val_loader, eval_interval=eval_interval, eval_max_batches=eval_max_batches)
        val_loss, val_acc, predictions, targets = eval_model(model, val_loader, device, loss_fn=loss_fn)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
    save_model(model, tokenizer, output_dir)
    fig = plot_metrics_history(history); fig.savefig(os.path.join(output_dir, "results", "training_history.png"))
    print(f"Saved training history to {os.path.join(output_dir, 'results', 'training_history.png')}")
    print(f"Saved model to {output_dir}")
    classical_dir = os.path.join(output_dir, "classical")
    print("\nTraining classical models...")
    
    print("="*50, "\nTRAINING CLASSICAL MODELS\n", "="*50)
    val_metrics = train_all_classical_models(train_df, classical_dir)
    with open(os.path.join(output_dir, "results", "classical_val_metrics.json"), 'w') as f:
        json.dump(val_metrics, f, indent=2)
    print("Classical training complete.")
    print(f"Saved classical metrics to {os.path.join(output_dir, 'results', 'classical_val_metrics.json')}")
    return {"best_val_acc": float(best_accuracy), "classical_val": val_metrics}
