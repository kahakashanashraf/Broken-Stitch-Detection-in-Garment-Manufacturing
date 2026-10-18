"""All-class filename provenance reconstruction. It is an inference, NOT manual visual verification."""
import argparse,zipfile,re,json,csv,collections,hashlib,random
from pathlib import Path
R=re.compile(r'^(?P<stem>(?:actual|faulty)_original_[A-Za-z0-9]+)\.(?:png|jpg|jpeg)_[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\.(?:png|jpg|jpeg)$',re.I)
NORMAL=re.compile(r'^actual_original_(?P<base>[A-Za-z]+)$',re.I)
FAULTY=re.compile(r'^faulty_original_(?P<base>[A-Za-z]+)(?P<n>\d+)$',re.I)

def run(zpath,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    entries=[];unparsed=[];family=collections.defaultdict(list)
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if not n.lower().endswith(('.png','.jpg','.jpeg')) or n.endswith('/'):continue
            m=R.fullmatch(Path(n).name)
            if not m:unparsed.append(n);continue
            stem=m.group('stem').lower()
            lab=stem.split('_')[0]
            family[(lab,stem)].append(n)
    normal_bases={m.group('base').lower():(lab,stem) for (lab,stem) in family if lab=='actual' if (m:=NORMAL.fullmatch(stem))}
    faulty_mappings=[];failed=[]
    for (lab,stem),names in sorted(family.items()):
        if lab!='faulty':continue
        m=FAULTY.fullmatch(stem)
        if not m or m.group('base').lower() not in normal_bases:
            failed.append(stem);continue
        faulty_mappings.append({'faulty_family':stem,'photo_parent_from_name':m.group('base').lower(),
                                'normal_family':normal_bases[m.group('base').lower()][1], 'augmented_image_count':len(names)})
    for (lab,stem),names in sorted(family.items()):
        m=NORMAL.fullmatch(stem) if lab=='actual' else FAULTY.fullmatch(stem)
        base=m.group('base').lower() if m else None
        if base is None or base not in normal_bases:continue
        for n in names:
            entries.append({'relative_path':n,'label':lab,'candidate_image_family':stem,
                            'inferred_photographed_source':'photo_candidate:'+base,
                            'original_zip_split':n.split('/')[1],
                            'photo_source_verified':'NO_FILENAME_AND_VISUAL_AUDIT_ONLY'})
    # stop on unmatched labels; do not hide silently
    if failed or unparsed or len(normal_bases)!=31 or len(faulty_mappings)!=151 or len(entries)!=7365:
        raise RuntimeError(f'Unexpected mapping. normals={len(normal_bases)} faulty_mapped={len(faulty_mappings)} unmatched={failed} unparsed={unparsed[:10]} images={len(entries)}')
    rng=random.Random(20261018)
    basekeys=sorted(normal_bases);rng.shuffle(basekeys)
    parts={base:'train' if i<21 else 'val' if i<26 else 'test' for i,base in enumerate(basekeys)}
    # only candidate-photo grouping; independent photographed provenance unverified
    for row in entries:
        base=row['inferred_photographed_source'].split(':',1)[1]
        row['split']=parts[base]
    with (out/'inferred_photo_parent_mapping.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(faulty_mappings[0]));w.writeheader();w.writerows(faulty_mappings)
    with (out/'inferred_photo_parent_split.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
    counts=collections.Counter((row['split'],row['label']) for row in entries)
    print('split counts',counts)
    cross={lab:collections.Counter(r['original_zip_split'] for r in entries if r['label']==lab) for lab in ('actual','faulty')}
    report={'inference_only':True,'all_augmented_images':len(entries),'normal_family_count':len(normal_bases),
            'faulty_photoshop_candidate_count':len(faulty_mappings),
            'inferred_underlying_photo_count':len(normal_bases),
            'inferred_parent_match_rate':len(faulty_mappings)/(len(faulty_mappings)+len(failed)),
            'filename_linking_rule':'faulty_original_XXN maps to actual_original_XX; N is Photoshop defect variant index',
            'photo_parent_disjoint_splits':dict(collections.Counter(parts.values())),
            'image_split_counts':{a+'_'+b:c for (a,b),c in counts.items()},
            'unmatched_groups':failed,'unparsed_files':unparsed,
            'photo_source_provenance_review':'Manual source photos / Photoshop edit history STILL REQUIRED. Filename rule might be wrong.',
            'top_visual_matches_in_earlier_report':'audits/photo_lineage_review/review_candidates_top5.json',
            'weights_trained_on_this_grouping':False,'external_test_accessed':False}
    (out/'photo_parent_filename_mapping_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.zip,a.out)