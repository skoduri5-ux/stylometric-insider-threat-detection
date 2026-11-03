"""Optional CERT R4.2 activity-log feature builder.

Expected input files (if available): logon.csv, device.csv, file.csv, email.csv, http.csv.
Produces one user-day row with activity aggregates. It does not infer malicious labels; merge
verified labels from the dataset's answer/ground-truth file separately.
"""
from pathlib import Path
import argparse
import pandas as pd

DATE_CANDIDATES = ['date', 'timestamp', 'datetime', 'time']
USER_CANDIDATES = ['user', 'username', 'user_id']

def first_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower: return lower[name.lower()]
    return None

def load_log(path, prefix):
    df = pd.read_csv(path, low_memory=False)
    user_col = first_col(df, USER_CANDIDATES)
    date_col = first_col(df, DATE_CANDIDATES)
    if user_col is None or date_col is None:
        print(f'SKIP {path.name}: cannot identify user/date columns; columns={list(df.columns)}')
        return None
    df = df.copy()
    df['_user'] = df[user_col].astype(str)
    df['_timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[df['_timestamp'].notna()]
    if df.empty: return None
    df['_day'] = df['_timestamp'].dt.floor('D')
    grp = df.groupby(['_user','_day'], dropna=False)
    out = grp.size().rename(f'{prefix}_count').to_frame()
    out[f'{prefix}_after_hours_count'] = grp['_timestamp'].apply(lambda s: int(((s.dt.hour < 8) | (s.dt.hour >= 18) | (s.dt.dayofweek >= 5)).sum()))
    out[f'{prefix}_active_hours'] = grp['_timestamp'].apply(lambda s: int(s.dt.hour.nunique()))
    # Common categorical fields in CERT logs are summarized only when present.
    for candidate, suffix in [('activity','activity_types'), ('url','unique_urls'), ('filename','unique_files'), ('to','unique_recipients')]:
        col = first_col(df, [candidate])
        if col:
            out[f'{prefix}_{suffix}'] = grp[col].nunique()
    out.index.names = ['username','date']
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/r42', help='Directory containing extracted CERT R4.2 CSV files')
    ap.add_argument('--out', default='data/r42_user_day_features.csv')
    args = ap.parse_args()
    root = Path(args.dir)
    if not root.exists(): raise SystemExit(f'Directory not found: {root}. Extract R4.2 files there first.')
    configs = [('logon.csv','logon'),('device.csv','device'),('file.csv','file'),('email.csv','email'),('http.csv','http')]
    merged = None; found = 0
    for filename, prefix in configs:
        path = root / filename
        if not path.exists(): continue
        found += 1
        part = load_log(path, prefix)
        if part is not None: merged = part if merged is None else merged.join(part, how='outer')
    if not found: raise SystemExit(f'No recognized log files found in {root}. Expected logon.csv, device.csv, file.csv, email.csv and/or http.csv.')
    if merged is None: raise SystemExit('No compatible user/date rows could be extracted. Inspect the printed column diagnostics.')
    merged = merged.fillna(0).reset_index()
    merged['date'] = pd.to_datetime(merged['date']).dt.strftime('%Y-%m-%d')
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); merged.to_csv(out, index=False)
    print(f'Wrote {len(merged)} user-day rows to {out}')
    print('NOTE: this output has no ground-truth label. Merge verified labels before supervised training/evaluation.')

if __name__ == '__main__': main()
