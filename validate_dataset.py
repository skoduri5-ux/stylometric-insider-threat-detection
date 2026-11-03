from pathlib import Path
import argparse, json
import pandas as pd

EXPECTED = ['email_id','timestamp','date','hour','day_of_week','is_weekend','is_after_hours','username','department','pc','from','to','cc','bcc','recipient_count','external_recipient','personal_email_recipient','subject','content','word_count','char_count','sentence_count','attachment_count','attachments','size','split','label','threat_scenario']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',default='data/emails_labeled_SYNTHETIC.csv'); a=p.parse_args()
    path=Path(a.csv); df=pd.read_csv(path)
    errors=[]
    if df.columns.tolist()!=EXPECTED: errors.append('Column names/order do not match the expected 28-column schema.')
    if len(df.columns)!=28: errors.append(f'Expected 28 columns, found {len(df.columns)}.')
    if 'label' in df and not set(df.label.dropna().astype(str).str.lower()).issubset({'normal','suspicious','0','1'}): errors.append('Unexpected label values.')
    if 'email_id' in df and df.email_id.duplicated().any(): errors.append(f'Duplicate email_id rows: {int(df.email_id.duplicated().sum())}')
    if 'content' in df and df.content.isna().any(): errors.append(f'Missing content rows: {int(df.content.isna().sum())}')
    summary={'path':str(path),'rows':len(df),'columns':len(df.columns),'label_counts':df.label.astype(str).str.lower().value_counts(dropna=False).to_dict() if 'label' in df else {},'split_label_counts':pd.crosstab(df.get('split',pd.Series('unspecified',index=df.index)),df.get('label',pd.Series('unknown',index=df.index))).to_dict(),'duplicate_email_ids':int(df.email_id.duplicated().sum()) if 'email_id' in df else None,'missing_content':int(df.content.isna().sum()) if 'content' in df else None,'errors':errors}
    print(json.dumps(summary,indent=2,default=str))
    if errors: raise SystemExit(1)
if __name__=='__main__': main()
