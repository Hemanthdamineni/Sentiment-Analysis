import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import re
import os

class AmazonReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length=128):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def load_data(file_path=None, use_huggingface=True, sample_size=None):
    def _clean_text(t):
        t = str(t).lower()
        t = re.sub(r"https?://\S+", " ", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    keywords = ["watch","smartwatch","fitbit","wearable","fitness band","smart watch","band","gear","amazfit","garmin","mi band","huawei watch"]
    pattern = r"\b(?:" + "|".join([re.escape(k) for k in keywords]) + r")\b"
    if use_huggingface:
        cache_path = os.path.join("backend/dataset", "smartwatch_reviews.csv")
        loaded_from_cache = False
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            loaded_from_cache = True
        else:
            from datasets import load_dataset
            dataset = load_dataset("mteb/amazon_reviews_multi", "en")
            parts = []
            for split in ["train","validation","test"]:
                if split in dataset:
                    tmp = pd.DataFrame(dataset[split])
                    tmp = tmp.rename(columns={"text":"text","label":"label"})
                    parts.append(tmp)
            df = pd.concat(parts, axis=0, ignore_index=True)
            df1 = df[df['text'].astype(str).str.contains(pattern, case=False, regex=True)]
            if len(df1) < 100:
                loose = "|".join([re.escape(k) for k in keywords])
                df1 = df[df['text'].astype(str).str.contains(loose, case=False, regex=True)]
            df = df1
            df['text'] = df['text'].apply(_clean_text)
            df = df[df['text'].str.len() > 0]
            df['sentiment'] = df['label'].apply(lambda x: 0 if x <= 1 else (1 if x == 2 else 2))
            os.makedirs("backend/dataset", exist_ok=True)
            df[['text','sentiment']].to_csv(cache_path, index=False)
        if loaded_from_cache:
            print(f"\nData source: local cache {cache_path}")
        else:
            print("\nData source: huggingface dataset")
        if len(df) == 0:
            raise ValueError("No smartwatch reviews found after filtering; try adjusting keywords or source dataset.")
        train_df, test_df = train_test_split(df[['text','sentiment']], test_size=0.2, random_state=42, stratify=df['sentiment'])
        if sample_size:
            train_df = train_df.sample(min(sample_size, len(train_df)), random_state=42)
            test_df = test_df.sample(min(max(1, sample_size//5), len(test_df)), random_state=42)
        print(f"\nTotal filtered: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}, Sample size: {sample_size}")
        train_counts = train_df['sentiment'].value_counts().to_dict()
        test_counts = test_df['sentiment'].value_counts().to_dict()
        print(f"Class distribution train={train_counts}, test={test_counts}")
        return train_df[['text','sentiment']], test_df[['text','sentiment']]
    else:
        if file_path is None:
            raise ValueError("File path must be provided when use_huggingface is False")
        df = pd.read_csv(file_path)
        df = df[df['text'].astype(str).str.contains(pattern, case=False, regex=True)]
        df['text'] = df['text'].apply(_clean_text)
        df = df[df['text'].str.len() > 0]
        df['sentiment'] = df['rating'].apply(lambda x: 0 if int(x) <= 2 else (1 if int(x) == 3 else 2))
        train_df, test_df = train_test_split(df[['text','sentiment']], test_size=0.2, random_state=42, stratify=df['sentiment'])
        print("\nData source: local CSV")
        print(f"\nTotal filtered: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")
        train_counts = train_df['sentiment'].value_counts().to_dict()
        test_counts = test_df['sentiment'].value_counts().to_dict()
        print(f"Class distribution train={train_counts}, test={test_counts}")
        return train_df[['text','sentiment']], test_df[['text','sentiment']]

def create_data_loaders(train_df, test_df, tokenizer, batch_size=16, max_length=128, use_weighted_sampler=False):
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    train_dataset = AmazonReviewDataset(train_df['text'].to_numpy(), train_df['sentiment'].to_numpy(), tokenizer, max_length)
    val_dataset = AmazonReviewDataset(val_df['text'].to_numpy(), val_df['sentiment'].to_numpy(), tokenizer, max_length)
    test_dataset = AmazonReviewDataset(test_df['text'].to_numpy(), test_df['sentiment'].to_numpy(), tokenizer, max_length)
    if use_weighted_sampler:
        import numpy as np, torch
        labels = train_df['sentiment'].to_numpy()
        uniq, counts = np.unique(labels, return_counts=True)
        class_weight = {c: round((len(labels) / (counts[i] * len(uniq))),3) for i, c in enumerate(uniq)}
        sample_weights = torch.DoubleTensor([class_weight[int(y)] for y in labels])
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=2)
        print(f"\nUsing WeightedRandomSampler with class_weight={class_weight}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Loaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}, batch_size={batch_size}, max_length={max_length}")
    return train_loader, val_loader, test_loader

def get_class_names():
    return ["Negative","Neutral","Positive"]
