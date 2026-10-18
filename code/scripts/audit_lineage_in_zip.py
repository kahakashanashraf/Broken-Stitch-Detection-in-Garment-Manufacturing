"""Candidate augmentation-family grouping from original ZIP metadata.
Not proof of image provenance: manual parent tracing remains required.
"""
import argparse,collections,json,re,zipfile
from pathlib import Path,PurePosixPath

UUID=re.compile(r'^(?P<stem>.+\.(?:png|jpg|jpeg))_(?P<uuid>[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\.(?:png|jpg|jpeg)$',re.I)
IMG={'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'}
def run(src,out):
    groups=collections.defaultdict(lambda:collections.Counter());counts=collections.Counter(); examples=collections.defaultdict(list)
    missing=[]; candidates=[]
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            if info.is_dir():continue
            p=PurePosixPath(info.filename)
            if p.suffix.lower() not in IMG:continue
            parts=p.parts
            split=next((c for c in parts if c.lower() in ('train','val','test')),None)
            label=next((c for c in parts if c.lower() in ('actual','faulty')),None)
            if split is None or label is None:
                missing.append(info.filename);continue
            m=UUID.fullmatch(p.name)
            if m:key=m.group('stem').lower();counts['uuid_augmentation_pattern']+=1
            else:key=p.name.lower();counts['nonuuid_name_pattern']+=1
            # NEVER infer these group names as verified originals; filename-only candidates
            groups[key][split]+=1;groups[key][label]+=1
            if len(examples[key])<3:examples[key].append(info.filename)
            candidates.append((info.filename,key,split,label))
    overlap=[];by_split=collections.Counter();by_label=collections.Counter()
    for key,n in groups.items():
        sp=[k for k in ('train','val','test') if n[k]]
        if len(sp)>1:
            overlap.append({'candidate_source':key,'split_counts':{s:n[s] for s in sp},'label_counts':{s:n[s] for s in ('actual','faulty')},'examples':examples[key]})
            for s in sp:by_split[s]+=n[s]
            for s in ('actual','faulty'):by_label[s]+=n[s]
    overlaps=sorted(overlap,key=lambda x:sum(x['split_counts'].values()),reverse=True)
    total=len(candidates)
    outobj={
        'images_with_recognized_split_and_label':total,'name_patterns':dict(counts),
        'candidate_parent_groups':len(groups),
        'candidate_parent_groups_crossing_splits':len(overlaps),
        'potential_leakage_images':sum(sum(g['split_counts'].values()) for g in overlaps),
        'potential_leakage_fraction':sum(sum(g['split_counts'].values()) for g in overlaps)/max(1,total),
        'potential_leakage_by_split':dict(by_split),'potential_leakage_by_label':dict(by_label),
        'potential_leakage_examples':overlaps[:50],
        'unrecognized_split_label_paths':missing[:30],
        'unique_candidate_groups_by_label':{s:sum(1 for g in groups.values() if g[s]) for s in ('actual','faulty')},
        'GO_NO_GO':'NO-GO original split if cross-split candidate parent matches; manual original lineage still required for any source-level accuracy claims',
        'interpretation':'Candidate parents inferred by stripping an appended UUID from each image filename. Same basename alone is not proof of identical photographed source.'
    }
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(outobj,indent=2)+'\n')
    print(json.dumps({k:v for k,v in outobj.items() if k!='potential_leakage_examples'},indent=2),flush=True)
    for i in overlaps[:10]:print('EXAMPLE',i,flush=True)
    return outobj
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--zip',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();run(a.zip,a.out)