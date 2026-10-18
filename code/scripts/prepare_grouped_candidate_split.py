"""PROVISIONAL family-disjoint split from UUID-suffixed public thesis images.

Not scientifically independent until Photoshop/source-photo parentage is verified.
Intentionally discards the original leaked 'train/val/test' assignments.
"""
import argparse,json,re
from pathlib import Path
from PIL import Image
import pandas as pd
from broken_stitch.data import create_group_split, audit_manifest

UUID=re.compile(r'^(?P<stem>.+\.(?:png|jpg|jpeg))_(?P<uuid>[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\.(?:png|jpg|jpeg)$',re.I)

def prepare(root,outdir,seed=42):
    root=Path(root).resolve(); out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    imgs=sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg') and p.is_file())
    rows=[]; errors=[]
    for p in imgs:
        rel=p.relative_to(root).as_posix(); comps=[x.lower() for x in p.relative_to(root).parts[:-1]]
        lab=set(comps)&{'actual','faulty'}
        if len(lab)!=1:raise ValueError(f'Ambiguous class in folder: {rel}')
        m=UUID.fullmatch(p.name)
        if m is None:raise ValueError(f'Expected known UUID-derived basename; cannot infer candidate family for: {rel}')
        try:
            with Image.open(p) as im: im.verify()
        except Exception as e:
            errors.append({'path':rel,'error':type(e).__name__});continue
        rows.append({'path':str(p),'relative_path':rel,'label':next(iter(lab)),
                     'source_id':'candidate:'+m.group('stem').lower(),
                     'is_synthetic':'unknown','split':''})
    df=pd.DataFrame(rows)
    if df.empty:raise ValueError('No valid images')
    if set(df.label)!={'actual','faulty'}:raise ValueError('Both classes required')
    inp=out/'candidate_groups_unsplit.csv';df.to_csv(inp,index=False)
    target=out/'candidate_group_splits.csv'
    split=create_group_split(str(inp),str(target),seed=seed,attempts=300)
    audited=audit_manifest(str(target),check_images=False,hash_all=True)
    summary={'label':'PROVISIONAL_CANDIDATE_FAMILY_DISJOINT_NOT_VERIFIED_SOURCE_INDEPENDENT',
             'images_in_archive':len(imgs),'images_valid':len(df),
             'corrupt_images':errors,'candidate_groups':df.source_id.nunique(),
             'class_counts':df.label.value_counts().to_dict(),
             'split_counts':split.groupby(['split','label']).size().to_dict(),
             'raw_existing_folders':'IGNORED',
             'known_risk':'Derivatives from a Photoshop-produced defective image may share an older original real photo with normal images; unresolved.',
             'audit':audited}
    summary['split_counts']={f'{a}/{b}':int(v) for (a,b),v in summary['split_counts'].items()}
    (out/'candidate_split_report.json').write_text(json.dumps(summary,indent=2,default=str))
    # Portable list, not absolute temporary runner paths
    split[['relative_path','label','source_id','is_synthetic','split']].to_csv(out/'candidate_group_splits_portable.csv',index=False)
    print(json.dumps({k:v for k,v in summary.items() if k!='audit'},indent=2),flush=True)
    return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--outdir',required=True)
    a=ap.parse_args();prepare(a.root,a.outdir)