import torch
import torch.nn as nn
from transformers import DistilBertModel
import numpy as np
from tqdm import tqdm

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout_p=0.4):
        super(SentimentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(self.distilbert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.drop(pooled_output)
        return self.out(output)

def train_epoch(model, data_loader, optimizer, device, scheduler=None, loss_fn=None, gradient_accumulation_steps=1, eval_loader=None, eval_interval=None, eval_max_batches=1):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=True)
    step_idx = 0
    last_val_acc = None
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets)

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == targets)
        total_predictions += targets.shape[0]
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        step_idx += 1
        if step_idx % max(1, gradient_accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        batch_acc = (preds == targets).float().mean().item()
        cumulative_acc = correct_predictions.item() / total_predictions
        lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else None
        if eval_loader is not None and eval_interval and step_idx % max(1, eval_interval) == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                bcount = 0
                for vbatch in eval_loader:
                    input_ids_v = vbatch['input_ids'].to(device)
                    attention_mask_v = vbatch['attention_mask'].to(device)
                    targets_v = vbatch['targets'].to(device)
                    outputs_v = model(input_ids=input_ids_v, attention_mask=attention_mask_v)
                    _, preds_v = torch.max(outputs_v, dim=1)
                    val_correct += torch.sum(preds_v == targets_v)
                    val_total += targets_v.shape[0]
                    bcount += 1
                    if bcount >= max(1, eval_max_batches):
                        break
            last_val_acc = val_correct.item() / max(1, val_total)
            model.train()
        postfix = {'loss': np.mean(losses), 'batch_acc': batch_acc, 'cumulative_acc': cumulative_acc}
        if lr is not None:
            postfix['lr'] = lr
        if last_val_acc is not None:
            postfix['val_acc'] = last_val_acc
        progress_bar.set_postfix(postfix)

    epoch_acc = correct_predictions.item() / total_predictions
    return np.mean(losses), epoch_acc

def eval_model(model, data_loader, device, loss_fn=None):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_targets = []

    progress_bar = tqdm(data_loader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if loss_fn is None:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            total_predictions += targets.shape[0]
            losses.append(loss.item())

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            progress_bar.set_postfix({'loss': np.mean(losses), 'accuracy': correct_predictions.item() / total_predictions})

    epoch_acc = correct_predictions.item() / total_predictions
    return np.mean(losses), epoch_acc, all_predictions, all_targets

def save_model(model, tokenizer, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
