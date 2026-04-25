import os
import warnings
warnings.filterwarnings("ignore", message=r"Found Intel OpenMP.*", category=RuntimeWarning)
import pickle
import numpy as np
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

RANDOM_STATE = 42

def get_algorithms():
    return {
        'logistic_regression': {'estimator': LogisticRegression(max_iter=2000, class_weight='balanced'), 'use_svd_smote': True},
        'linear_svc': {'estimator': LinearSVC(class_weight='balanced'), 'use_svd_smote': True},
        'svc_rbf': {'estimator': SVC(kernel='rbf', probability=True, class_weight='balanced'), 'use_svd_smote': True},
        'sgd_classifier': {'estimator': SGDClassifier(random_state=RANDOM_STATE, class_weight='balanced'), 'use_svd_smote': True},
        'random_forest': {'estimator': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced'), 'use_svd_smote': True},
        'knn': {'estimator': KNeighborsClassifier(n_neighbors=5), 'use_svd_smote': True},
        'multinomial_nb': {'estimator': MultinomialNB(), 'use_svd_smote': False},
        'bernoulli_nb': {'estimator': BernoulliNB(), 'use_svd_smote': False},
    }

def train_all_classical_models(train_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X = train_df['text'].values; y = train_df['sentiment'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vec.fit_transform(X_train); X_val_vec = vec.transform(X_val)
    svd = TruncatedSVD(n_components=300, random_state=RANDOM_STATE)
    X_train_svd = svd.fit_transform(X_train_vec); X_val_svd = svd.transform(X_val_vec)
    counts = Counter(y_train); min_ratio = min(counts.values()) / float(sum(counts.values()))
    X_train_svd_bal, y_train_bal = X_train_svd, y_train
    if min_ratio < 0.2:
        try:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_svd_bal, y_train_bal = smote.fit_resample(X_train_svd, y_train)
        except Exception:
            X_train_svd_bal, y_train_bal = X_train_svd, y_train
    val_metrics = {}
    print(f"Training {len(get_algorithms())} classical algorithms")
    for name, cfg in tqdm(get_algorithms().items(), desc="Classical Training", leave=True):
        est = cfg['estimator']
        if cfg['use_svd_smote']:
            est.fit(X_train_svd_bal, y_train_bal); preds = est.predict(X_val_svd)
        else:
            est.fit(X_train_vec, y_train); preds = est.predict(X_val_vec)
        acc = accuracy_score(y_val, preds); pr, re, f1, _ = precision_recall_fscore_support(y_val, preds, average='weighted', zero_division=0)
        val_metrics[name] = {'accuracy': float(acc), 'precision': float(pr), 'recall': float(re), 'f1': float(f1)}
        payload = {'vec': vec, 'svd': svd if cfg['use_svd_smote'] else None, 'model': est, 'name': name}
        with open(os.path.join(output_dir, f"{name}.pkl"), 'wb') as f: pickle.dump(payload, f)
    return val_metrics

def load_all_classical_models(output_dir):
    models = {}
    for fname in os.listdir(output_dir):
        if fname.endswith('.pkl'):
            with open(os.path.join(output_dir, fname), 'rb') as f:
                obj = pickle.load(f)
                if isinstance(obj, dict): models[fname[:-4]] = obj
                elif isinstance(obj, tuple) and len(obj) == 2: models[fname[:-4]] = {'vec': obj[0], 'svd': None, 'model': obj[1]}
    return models

def evaluate_all_classical_models(models, test_df):
    X = test_df['text'].values; y = test_df['sentiment'].values; results = {}
    for name, bundle in models.items():
        vec = bundle['vec']; svd = bundle.get('svd'); model = bundle['model']
        X_vec = vec.transform(X); X_vec = svd.transform(X_vec) if svd is not None else X_vec
        preds = model.predict(X_vec); acc = accuracy_score(y, preds); pr, re, f1, _ = precision_recall_fscore_support(y, preds, average='weighted', zero_division=0)
        entry = {'accuracy': float(acc), 'precision': float(pr), 'recall': float(re), 'f1': float(f1)}
        try:
            Y_bin = label_binarize(y, classes=sorted(np.unique(y)))
            if hasattr(model, 'decision_function'): scores = model.decision_function(X_vec)
            elif hasattr(model, 'predict_proba'): scores = model.predict_proba(X_vec)
            else: scores = None
            if scores is not None:
                auc_macro = roc_auc_score(Y_bin, scores, average='macro', multi_class='ovr'); entry['roc_auc_macro'] = float(auc_macro)
        except Exception: pass
        results[name] = entry
    return results

def plot_and_save_classical_rocs(models, test_df, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt, os
    os.makedirs(out_dir, exist_ok=True)
    X = test_df['text'].values; y = test_df['sentiment'].values; classes = sorted(np.unique(y))
    Y_bin = label_binarize(y, classes=classes); class_names = ['Negative','Neutral','Positive']
    for name, bundle in models.items():
        vec = bundle['vec']; svd = bundle.get('svd'); model = bundle['model']
        X_vec = vec.transform(X); X_vec = svd.transform(X_vec) if svd is not None else X_vec
        if hasattr(model, 'decision_function'): scores = model.decision_function(X_vec)
        elif hasattr(model, 'predict_proba'): scores = model.predict_proba(X_vec)
        else: continue
        plt.figure(figsize=(8,6))
        for i, cname in zip(range(len(classes)), class_names[:len(classes)]):
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(Y_bin[:, i], scores[:, i]); plt.plot(fpr, tpr, label=f"{cname}")
            except Exception: continue
        plt.plot([0,1],[0,1],'k--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'ROC Curves – {name}'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'roc_{name}.png')); plt.close()

def save_wordclouds(train_df, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True); stopwords = set(STOPWORDS)
    for label_id, label_name in zip([0,1,2], ['Negative','Neutral','Positive']):
        texts = train_df[train_df['sentiment']==label_id]['text']
        if len(texts) == 0: continue
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(" ".join(texts))
        wc.to_file(os.path.join(out_dir, f'wordcloud_{label_name.lower()}.png'))
