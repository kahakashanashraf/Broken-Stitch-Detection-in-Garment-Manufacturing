"""OPTIONAL naming-rule helper. Output MUST be reviewed against original images.

Example: --regex 'parent(?P<source>\\d+)' links parent042_aug01 to parent042.
Do not infer source IDs from random augmentation filenames.
"""
import argparse,re
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--inventory',required=True)
p.add_argument('--out',required=True)
p.add_argument('--regex',required=True,help='Regex with named capture (?P<source>...) applied to relative path')
a=p.parse_args()
pattern=re.compile(a.regex)
if 'source' not in pattern.groupindex:raise SystemExit('Regex must include named capture (?P<source>...)')
df=pd.read_csv(a.inventory,dtype=str,keep_default_na=False)
if 'relative_path' not in df:raise SystemExit('Inventory must contain relative_path')
assigned=[]
for path in df.relative_path:
    match=pattern.search(path)
    assigned.append(match.group('source') if match else '')
df['source_id']=assigned
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
df.to_csv(a.out,index=False)
print(f'Assigned: {sum(bool(x) for x in assigned)}/{len(assigned)}; remaining blanks BLOCK splitting/training')
print('WARNING: Filename-based lineage is a guess until verified against source/edited parent images.')