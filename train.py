from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_fscore_support,
                             average_precision_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'; ART.mkdir(exist_ok=True)


def truthy(s):
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors='coerce').fillna(0)
    return s.astype(str).str.strip().str.lower().map({'true':1,'yes':1,'1':1,'false':0,'no':0,'0':0}).fillna(0)


def normalize_label(s):
    v = s.astype(str).str.strip().str.lower()
    mapped = v.map({'normal':0,'suspicious':1,'0':0,'1':1,'false':0,'true':1})
    if mapped.isna().any():
        bad = sorted(v[mapped.isna()].unique().tolist())
        raise SystemExit(f'Unsupported label values: {bad}. Expected normal/suspicious or 0/1.')
    return mapped.astype(int)


def text_join(df):
    subject = df.get('subject', pd.Series('', index=df.index)).fillna('').astype(str)
    content = df.get('content', pd.Series('', index=df.index)).fillna('').astype(str)
    return (subject + ' ' + content).tolist()


def stylometric_frame(df):
    text = df.get('content', pd.Series('', index=df.index)).fillna('').astype(str)
    subject = df.get('subject', pd.Series('', index=df.index)).fillna('').astype(str)
    words = text.map(lambda x: re.findall(r"\b\w+\b", x))
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
    out['subject_length'] = subject.str.len()
    # Intentionally exclude recipient/attachment/size metadata from the stylometry model.
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def temporal_frame(df):
    out = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df.get('timestamp', pd.Series(pd.NaT, index=df.index)), errors='coerce')
    hour_col = pd.to_numeric(df.get('hour', pd.Series(np.nan, index=df.index)), errors='coerce')
    out['hour'] = hour_col.fillna(ts.dt.hour).fillna(12)
    dow_col = df.get('day_of_week', pd.Series('', index=df.index)).astype(str).str.lower()
    dow_map = {'monday':0,'tuesday':1,'wednesday':2,'thursday':3,'friday':4,'saturday':5,'sunday':6}
    dow_num = dow_col.map(dow_map).fillna(ts.dt.dayofweek).fillna(0)
    out['day_of_week_num'] = dow_num
    weekend_col = truthy(df.get('is_weekend', pd.Series(0, index=df.index)))
    out['is_weekend'] = weekend_col
    after_col = truthy(df.get('is_after_hours', pd.Series(0, index=df.index)))
    out['is_after_hours'] = after_col
    for c in ['recipient_count','attachment_count','size','word_count','char_count','sentence_count']:
        out[c] = pd.to_numeric(df[c], errors='coerce') if c in df else 0
    # Excludes external/personal recipient flags because in this synthetic dataset they nearly determine the label.
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def evaluate(y_true, pred, score):
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
    result = {'precision':float(p),'recall':float(r),'f1':float(f),
              'average_precision_pr_auc':float(average_precision_score(y_true, score)),
              'confusion_matrix_labels_0_1':confusion_matrix(y_true,pred,labels=[0,1]).tolist(),
              'classification_report':classification_report(y_true,pred,zero_division=0,output_dict=True)}
    if len(np.unique(y_true)) == 2: result['roc_auc'] = float(roc_auc_score(y_true, score))
    return result


def body_groups(df):
    # Group by normalized body to keep repeated content out of separate partitions.
    body = df.get('content', pd.Series('', index=df.index)).fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    # Include subject when body is empty so unrelated empty-body messages do not become one giant group.
    subj = df.get('subject', pd.Series('', index=df.index)).fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    return np.where(body.eq(''), 'EMPTY::' + subj, 'BODY::' + body)


def group_split(df, seed=42):
    groups = body_groups(df)
    # Hold out 15% of distinct body groups for final test.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_i, test_i = next(outer.split(df, df['label'], groups))
    trainval = df.iloc[trainval_i].copy(); test = df.iloc[test_i].copy()
    # Hold out ~17.65% of remaining rows (~15% overall) for validation.
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed+1)
    tr_rel, val_rel = next(inner.split(trainval, trainval['label'], body_groups(trainval)))
    train = trainval.iloc[tr_rel].copy(); val = trainval.iloc[val_rel].copy()
    # Fail safe if grouped split leaves a partition without both labels.
    for name, part in [('train',train),('validation',val),('test',test)]:
        if part['label'].nunique() < 2:
            raise SystemExit(f'Group-aware {name} split has only one class. Need more independent examples or a different group strategy.')
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description='Train temporal and non-temporal insider-threat ML baselines.')
    ap.add_argument('--csv', default=str(ROOT / 'data' / 'emails.csv'))
    args = ap.parse_args(); path = Path(args.csv)
    if not path.exists(): raise SystemExit(f'CSV not found: {path}. Pass --csv "path/to/emails_labeled_SYNTHETIC.csv"')
    df = pd.read_csv(path).reset_index(drop=True)
    required = {'label','content','subject'}
    missing = required - set(df.columns)
    if missing: raise SystemExit(f'Missing required columns: {sorted(missing)}')
    df['label'] = normalize_label(df['label'])
    df = df[df['content'].notna()].copy().reset_index(drop=True)
    if df['label'].nunique()!=2: raise SystemExit('Both normal and suspicious labels are required.')
    original = {}
    if 'split' in df:
        for s, part in df.groupby('split', dropna=False):
            original[str(s)]={'rows':int(len(part)),'class_counts':{str(k):int(v) for k,v in part['label'].value_counts().sort_index().items()}}
    groups = body_groups(df)
    grouped_label_conflicts = int(pd.DataFrame({'g':groups,'y':df['label']}).groupby('g').y.nunique().gt(1).sum())
    train, val, test = group_split(df)
    y_train=train.label.to_numpy(); y_val=val.label.to_numpy(); y_test=test.label.to_numpy()
    report={'dataset':str(path),'dataset_type':'synthetic; metrics measure performance on generated patterns, not real-world threat-detection validity',
      'rows_used':int(len(df)),'class_counts':{('normal' if k==0 else 'suspicious'):int(v) for k,v in df.label.value_counts().sort_index().items()},
      'original_split_diagnostic_only':original,'grouping':{'method':'normalized email body; empty bodies grouped by normalized subject','unique_groups':int(pd.Series(groups).nunique()),'duplicate_body_rows_beyond_unique':int(len(df)-pd.Series(groups).nunique()),'groups_with_conflicting_labels':grouped_label_conflicts,'group_overlap_between_train_validation_test':False},
      'development_split':{n:{'rows':int(len(d)),'class_counts':{('normal' if k==0 else 'suspicious'):int(v) for k,v in d.label.value_counts().sort_index().items()}} for n,d in [('train',train),('validation',val),('test',test)]},
      'leakage_controls':['Group split keeps normalized repeated email bodies within one partition.','Temporal model excludes external_recipient and personal_email_recipient because these nearly perfectly encode labels in this synthetic file.','Stylometry model uses subject/content TF-IDF and linguistic features only; no label/scenario/recipient metadata.'],
      'warnings':['Synthetic data: metrics are not evidence of real-world performance.','Check template and phrase leakage before reporting results.','Validation is held out; the final test set is used once for reported metrics.']}
    # Temporal model uses only temporal/activity numeric features.
    Xt_train=temporal_frame(train); Xt_val=temporal_frame(val); Xt_test=temporal_frame(test)
    temporal_model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42))
    temporal_model.fit(Xt_train,y_train)
    tv_score=temporal_model.predict_proba(Xt_val)[:,1]; tv_pred=(tv_score>=0.5).astype(int)
    tt_score=temporal_model.predict_proba(Xt_test)[:,1]; tt_pred=(tt_score>=0.5).astype(int)
    report['temporal_supervised']={'validation':evaluate(y_val,tv_pred,tv_score),'test':evaluate(y_test,tt_pred,tt_score),'features':list(Xt_train.columns)}
    temporal_if=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),IsolationForest(n_estimators=300,contamination='auto',random_state=42,n_jobs=-1))
    temporal_if.fit(Xt_train.loc[y_train==0])
    ta=-temporal_if.decision_function(Xt_test)
    report['temporal_anomaly_detector']={'normal_train_rows':int((y_train==0).sum()),'test_mean_anomaly_score_normal':float(ta[y_test==0].mean()),'test_mean_anomaly_score_suspicious':float(ta[y_test==1].mean()),'note':'Uncalibrated anomaly ranking, not a probability.'}
    # Non-temporal text + style model.
    text_features=FeatureUnion([('word_tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,strip_accents='unicode')),('char_tfidf',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=60000,sublinear_tf=True,strip_accents='unicode'))])
    tr=train.copy(); va=val.copy(); te=test.copy()
    for d in (tr,va,te):
        d['_text_for_model']=text_join(d)
        st=stylometric_frame(d)
        for c in st.columns: d[c]=st[c]
    style_cols=list(stylometric_frame(train).columns)
    preprocess=ColumnTransformer([('text',text_features,'_text_for_model'),('style',make_pipeline(SimpleImputer(strategy='median'),StandardScaler(with_mean=False)),style_cols)],remainder='drop',sparse_threshold=0.3)
    non_temporal_model=Pipeline([('features',preprocess),('classifier',LogisticRegression(class_weight='balanced',max_iter=2000,C=1.0,random_state=42))])
    non_temporal_model.fit(tr,y_train)
    nv=non_temporal_model.predict_proba(va)[:,1]; npred=(nv>=0.5).astype(int)
    ns=non_temporal_model.predict_proba(te)[:,1]; nsp=(ns>=0.5).astype(int)
    report['non_temporal_supervised']={'validation':evaluate(y_val,npred,nv),'test':evaluate(y_test,nsp,ns),'features':{'text':'subject + content word/character TF-IDF','stylometric_numeric':style_cols}}
    style_train=stylometric_frame(train); style_test=stylometric_frame(test)
    style_if=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),IsolationForest(n_estimators=300,contamination='auto',random_state=42,n_jobs=-1))
    style_if.fit(style_train.loc[y_train==0]); sa=-style_if.decision_function(style_test)
    report['non_temporal_anomaly_detector']={'normal_train_rows':int((y_train==0).sum()),'test_mean_anomaly_score_normal':float(sa[y_test==0].mean()),'test_mean_anomaly_score_suspicious':float(sa[y_test==1].mean()),'note':'Uncalibrated anomaly ranking, not a probability.'}
    joblib.dump({'model':temporal_model,'anomaly_model':temporal_if,'feature_columns':list(Xt_train.columns)},ART/'temporal_model.joblib')
    joblib.dump({'model':non_temporal_model,'anomaly_model':style_if,'style_columns':style_cols},ART/'non_temporal_model.joblib')
    (ART/'metrics.json').write_text(json.dumps(report,indent=2,default=float))
    print(json.dumps({'rows_used':report['rows_used'],'class_counts':report['class_counts'],'development_split':report['development_split'],'temporal_test':report['temporal_supervised']['test'],'non_temporal_test':report['non_temporal_supervised']['test'],'grouping':report['grouping']},indent=2))
    print(f'\nSaved model artifacts and metrics to {ART}')

if __name__=='__main__': main()
