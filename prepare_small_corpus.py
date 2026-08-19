"""Prepare the instructor-provided first-5000 evidence scope for the notebook schema."""
import ast, json, re
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent
src=root/'data'/'graphrag_golden_50_first5000_detailed.csv'
dst=root/'data'/'hackernoon_subset.csv'
df=pd.read_csv(src)
rows=[]
for r in df.itertuples(index=False):
    evidence=str(r.reference_evidence)
    dates=re.findall(r'\((\d{4}-\d{2}-\d{2})', evidence)
    title=evidence.split(': ',1)[-1].split(' | ',1)[0]
    text=' '.join([str(r.reference_answer), evidence, str(r.gold_reasoning), str(r.scoring_notes)])
    rows.append({'id':r.id,'title':title,'text':text,'published_date':dates[0] if dates else '2023-01-01','url':str(r.evidence_urls_json)})
pd.DataFrame(rows).to_csv(dst,index=False)
print(f'created {dst} rows={len(rows)}')
