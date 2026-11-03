from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
    precision_recall_fscore_support, average_precision_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'


def normalize_label(s):
    v = s.astype(str).str.strip().str.lower()
    mapped = v.map({'normal': 0, 'suspicious': 1, '0': 0, '1': 1,
                    'false': 0, 'true': 1})
    if mapped.isna().any():
        raise ValueError(f"Unsupported labels: {sorted(v[mapped.isna()].unique())}")
    return mapped.astype(int)


def stylometric_frame(df):
    text = df['content'].fillna('').astype(str)
    words = text.map(lambda x: re.findall(r'\b\w+\b', x))
    out = pd.DataFrame(index=df.index)
    out['word_count_calc'] = words.map(len)
    out['char_count_calc'] = text.str.len()
    out['sentence_count_calc'] = text.map(lambda x: len(re.findall(r'[.!?]+', x)) if x.strip() else 0)
    out['avg_word_len'] = words.map(lambda ws: float(np.mean([len(w) for w in ws])) if ws else 0)
    out['type_token_ratio'] = words.map(lambda ws: len(set(w.lower() for w in ws)) / max(1, len(ws)))
    out['uppercase_ratio'] = text.map(lambda x: sum(c.isupper() for c in x) / max(1, sum(c.isalpha() for c in x)))
    out['punctuation_count'] = text.map(lambda x: sum(not c.isalnum() and not c.isspace() for c in x))
    out['exclamation_count'] = text.str.count('!')
    out['question_count'] = text.str.count(r'\?')
    out['comma_count'] = text.str.count(',')
    out['newline_count'] = text.str.count(r'\n')
    out['avg_sentence_chars'] = out['char_count_calc'] / out['sentence_count_calc'].clip(lower=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def groups_for(df):
    body = df['content'].fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    subject = df['subject'].fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    return np.where(body.eq(''), 'EMPTY::' + subject, 'BODY::' + body)


def split_groups(df, seed=42):
    groups = groups_for(df)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, df.label, groups))
    trainval, test = df.iloc[trainval_idx].copy(), df.iloc[test_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed + 1)
    tr_idx, val_idx = next(inner.split(trainval, trainval.label, groups_for(trainval)))
    train, val = trainval.iloc[tr_idx].copy(), trainval.iloc[val_idx].copy()
    for name, part in [('train', train), ('validation', val), ('test', test)]:
        if part.label.nunique() != 2:
            raise ValueError(f'{name} partition does not contain both labels; try another --seed.')
    return train, val, test


def make_features():
    return FeatureUnion([
        ('word_tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2,
            max_features=60000, sublinear_tf=True, strip_accents='unicode')),
        ('char_tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
            min_df=2, max_features=60000, sublinear_tf=True, strip_accents='unicode'))
    ])


def prepare(part, style_cols):
    d = part.copy()
    d['_text_for_model'] = d.content.fillna('').astype(str)
    st = stylometric_frame(d)
    for col in style_cols:
        d[col] = st[col]
    return d


def choose_threshold(y, scores):
    # Select threshold using validation data only; maximize F1.
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best = (0.5, -1.0)
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
        if f1 > best[1]:
            best = (float(threshold), float(f1))
    return best


def evaluate(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        'threshold': float(threshold), 'precision': float(p), 'recall': float(r), 'f1': float(f1),
        'accuracy': float((tn + tp) / max(1, tn + fp + fn + tp)),
        'false_positive_rate': float(fp / max(1, fp + tn)),
        'confusion_matrix_labels_0_1': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'classification_report': classification_report(y, pred, zero_division=0, output_dict=True),
        'pr_auc_average_precision': float(average_precision_score(y, scores)),
        'roc_auc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Non-temporal model comparison: text-only vs text + stylometry.')
    parser.add_argument('--csv', required=True, help='Path to labelled CSV with content, subject and label columns')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    path = Path(args.csv).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f'CSV not found: {path}')
    df = pd.read_csv(path)
    required = {'content', 'subject', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'Missing required columns: {sorted(missing)}')
    df = df[df.content.notna()].copy().reset_index(drop=True)
    df['label'] = normalize_label(df.label)
    if df.label.nunique() != 2:
        raise SystemExit('Dataset must contain both normal and suspicious labels.')

    train, val, test = split_groups(df, args.seed)
    y_train, y_val, y_test = train.label.to_numpy(), val.label.to_numpy(), test.label.to_numpy()
    style_cols = list(stylometric_frame(train).columns)
    tr, va, te = [prepare(x, style_cols) for x in (train, val, test)]

    text_only = ColumnTransformer([
        ('text', make_features(), '_text_for_model')
    ], remainder='drop', sparse_threshold=0.3)
    combined = ColumnTransformer([
        ('text', make_features(), '_text_for_model'),
        ('style', Pipeline([('imputer', SimpleImputer(strategy='median')),
                            ('scale', StandardScaler(with_mean=False))]), style_cols)
    ], remainder='drop', sparse_threshold=0.3)

    models = {
        'text_only': Pipeline([('features', text_only), ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=2000, C=1.0, random_state=args.seed))]),
        'text_plus_stylometry': Pipeline([('features', combined), ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=2000, C=1.0, random_state=args.seed))])
    }

    ART.mkdir(parents=True, exist_ok=True)
    report = {
        'dataset': str(path),
        'warning': 'Synthetic-data metrics do not establish real-world insider-threat detection performance.',
        'split_method': 'GroupShuffleSplit on normalized email body; identical bodies stay in one partition.',
        'rows': int(len(df)),
        'class_counts': {('normal' if int(k) == 0 else 'suspicious'): int(v) for k, v in df.label.value_counts().items()},
        'partitions': {name: {'rows': int(len(part)), 'class_counts': {('normal' if int(k)==0 else 'suspicious'): int(v) for k,v in part.label.value_counts().items()}}
                       for name, part in [('train', train), ('validation', val), ('test', test)]},
        'models': {}
    }

    for name, model in models.items():
        model.fit(tr, y_train)
        val_scores = model.predict_proba(va)[:, 1]
        threshold, val_best_f1 = choose_threshold(y_val, val_scores)
        test_scores = model.predict_proba(te)[:, 1]
        report['models'][name] = {
            'validation_best_f1': val_best_f1,
            'validation': evaluate(y_val, val_scores, threshold),
            'test': evaluate(y_test, test_scores, threshold),
            'threshold_selected_on_validation_only': threshold
        }
        joblib.dump({'model': model, 'threshold': threshold,
                     'style_columns': style_cols if name == 'text_plus_stylometry' else [],
                     'model_type': name, 'text_field': 'content only'}, ART / f'non_temporal_{name}_v2.joblib')

    (ART / 'non_temporal_v2_metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('\nSaved separate v2 models and metrics in:', ART)
    print('The existing non_temporal_model.joblib was not changed.')

if __name__ == '__main__':
    main()
