"""Materialize a group-exclusive research dataset from the original broken-stitch ZIP.

The grouped source IDs are inferred from filenames, NOT independently verified photos.
This is a *data partition* repair; it does not reconstruct original unaugmented photos.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

EXPECTED_ZIP_BYTES=2146784353
EXPECTED_ZIP_SHA256='e7aed1355a712c14bbe6560fefbb3dfab546a38970f836d129f5721e0b046979'
EXPECTED_MANIFEST_SHA256='769d10dc6f6651f24d655ae0ed6bc1f780a50f24882c0b7fd5073fec1a51d2cb'
BAD='dataset/train/actual/actual_original_AB.png_579370ba-65d0-4793-a3c0-0f73fbf6c367.png'
STEMS=re.compile(r'^(actual|faulty)_original_([a-z]+)([0-9]+)?$',re.I)
SPLITS=('train','val','test')
LABELS=('actual','faulty')

def filehash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def family_photo_base(family,label):
    m=STEMS.fullmatch(family)
    if not m or m.group(1).lower()!=label:
        raise ValueError('Unsupported family/class '+repr((family,label)))
    if (label=='actual' and m.group(3)) or (label=='faulty' and not m.group(3)):
        raise ValueError('Unexpected source filename pattern '+family)
    return m.group(2).lower()

def validate_group_splits(rows):
    seen_paths=set()
    parents={}
    labels_by_split=Counter()
    fams=defaultdict(set)
    for r in rows:
        old=r['relative_path']
        if old in seen_paths:raise ValueError('Duplicate input manifest file '+old)
        seen_paths.add(old)
        sp=r['split'];label=r['label'];parent=r['inferred_photographed_source']
        if sp not in SPLITS or label not in LABELS:raise ValueError('Invalid class/split')
        if parent!='photo_candidate:'+family_photo_base(r['candidate_image_family'],label):
            raise ValueError('Source-root name inconsistent: '+old)
        if parent in parents and parents[parent]!=sp:
            raise ValueError('Photo parent crosses splits '+parent)
        parents[parent]=sp
        fams[(sp,label)].add(r['candidate_image_family'])
        labels_by_split[(sp,label)]+=1
    if Counter(parents.values())!=Counter({'train':21,'val':5,'test':5}):
        raise ValueError('Expected 31 parent roots and 21/5/5 split')
    if labels_by_split!=Counter({('train','actual'):1878,('train','faulty'):3141,
                                ('val','actual'):433,('val','faulty'):826,
                                ('test','actual'):422,('test','faulty'):664}):
        raise ValueError('Unexpected non-corrupt partition counts '+repr(labels_by_split))
    if sum(len(s) for s in fams.values())!=182:
        raise ValueError('Unexpected candidate image edit family count')
    return parents,labels_by_split

def main(src_path, manifest_path, target):
    src_path=Path(src_path);manifest_path=Path(manifest_path);target=Path(target)
    target.mkdir(parents=True,exist_ok=True)
    if src_path.stat().st_size!=EXPECTED_ZIP_BYTES or filehash(src_path)!=EXPECTED_ZIP_SHA256:
        raise ValueError('STOP: source ZIP identity mismatch')
    raw=manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_MANIFEST_SHA256:
        raise ValueError('STOP: photo-parent manifest changed')
    full=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(full)!=7365:raise ValueError('STOP: unexpected original image count')
    excluded=[r for r in full if r['relative_path']==BAD]
    if len(excluded)!=1:raise ValueError('STOP: corrupt image record not found exactly once')
    rows=[r for r in full if r['relative_path']!=BAD]
    parents,counts=validate_group_splits(rows)
    seen_hash={}
    exported=[]
    reps={}
    archive_files={s:target/(s+'.zip') for s in SPLITS}
    archive_hashes={}
    with zipfile.ZipFile(src_path,'r') as original:
        keys=set(original.namelist())
        if set(r['relative_path'] for r in full)-keys:
            raise ValueError('Missing some source images from original ZIP')
        with (zipfile.ZipFile(archive_files['train'],'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as tz,
              zipfile.ZipFile(archive_files['val'],'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as vz,
              zipfile.ZipFile(archive_files['test'],'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as ez):
            dest={'train':tz,'val':vz,'test':ez}
            used=set()
            for ix,r in enumerate(sorted(rows,key=lambda x:(x['split'],x['label'],x['relative_path']))):
                old=r['relative_path']
                if '..' in Path(old).parts or not old.startswith('dataset/'):
                    raise ValueError('Invalid input member path')
                sp=r['split'];label=r['label'];base=Path(old).name
                new=f'{sp}/{label}/{base}'
                if new in used:raise ValueError('ZIP output name collision '+new)
                used.add(new)
                b=original.read(old) # CRC checked by zipfile on read
                with Image.open(io.BytesIO(b)) as im:
                    im.verify() # fail closed if unexpected other corrupt image
                content_sha=hashlib.sha256(b).hexdigest()
                if content_sha in seen_hash and seen_hash[content_sha]!=sp:
                    raise ValueError('Exact bytes duplicated across splits')
                seen_hash[content_sha]=sp
                dest[sp].writestr(new,b)
                ent=dict(r)
                ent.update({'clean_path':new,'sha256':content_sha,'bytes':len(b),
                            'data_status':'AUGMENTED_SOURCE_IMAGE_NOT_CONFIRMED_ORIGINAL'})
                exported.append(ent)
                key=(sp,r['candidate_image_family'])
                if key not in reps:
                    reps[key]={'split':sp,'candidate_image_family':r['candidate_image_family'],
                               'inferred_photographed_source':r['inferred_photographed_source'],
                               'label':label,'clean_path':new,'warning':'EXAMPLE_AUGMENTED_IMAGE_NOT_RAW_SOURCE'}
                if ix%1000==0: print('copied',ix,'/',len(rows),flush=True)
    for sp in SPLITS:
        p=archive_files[sp]
        if p.stat().st_size>=2_147_483_648:
            raise ValueError('Release asset too large for GitHub: '+str(p))
        with zipfile.ZipFile(p,'r') as z:
            if z.testzip() is not None:raise ValueError('Corrupt output archive: '+sp)
            if len(z.namelist())!=sum(k[0]==sp for k in counts for _ in range(counts[k])):
                raise ValueError('Output ZIP count mismatch')
        archive_hashes[sp]={'name':p.name,'bytes':p.stat().st_size,'sha256':filehash(p)}
    cols=list(exported[0])
    with (target/'clean_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(exported)
    with (target/'evaluation_representative_candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(next(iter(reps.values()))));w.writeheader();w.writerows(reps.values())
    audit={
      'version':'v2_inferred_photo_parent_exclusive_NOT_HUMAN_VERIFIED',
      'source_archive_sha256':EXPECTED_ZIP_SHA256,
      'source_manifest_sha256':EXPECTED_MANIFEST_SHA256,
      'original_images':7365,'quarantined_unreadable_image':BAD,'included_images':len(exported),
      'parents_total':len(parents),'parents_by_split':dict(Counter(parents.values())),
      'images_by_split_label':{sp:{lab:counts[(sp,lab)] for lab in LABELS} for sp in SPLITS},
      'filename_families_total':182,'exact_cross_split_byte_duplicates':0,
      'inferred_photo_parent_cross_split_overlaps':0,
      'representative_candidate_rows':len(reps),'archives':archive_hashes,
      'image_condition':'These images are previously augmented/Photoshop-derived; original raw photographs not recovered.',
      'test_condition':'Previously scored holdout; not a fresh untouched test after these results were inspected.',
      'scientific_gate':'NO GO for confirmed original-photo provenance or genuine independent camera test.'
    }
    (target/'quality_report.json').write_text(json.dumps(audit,indent=2))
    (target/'README.txt').write_text(
       'BROKEN STITCH DATASET v2 — PHOTO-PARENT-GROUPED, INFERRED, NOT VERIFIED\\n'
       'Extract train.zip, val.zip, test.zip to the SAME new directory.\\n'
       'Folders: train/actual, train/faulty, val/actual, val/faulty, test/actual, test/faulty.\\n'
       'Old leaky split assignments are ignored. Exactly one corrupt file excluded.\\n'
       'All 7364 images are historical augmented/Photoshop-derived examples, not recovered raw photos.\\n'
       'The 31 photo-root grouping is inferred from names and visual matches, NOT verified source records.\\n'
       'Test group was previously evaluated and must not be reused as untouched after tuning.\\n'
       'Evaluation_representative_candidates.csv contains one previously augmented sample per EDIT family; NOT unaugmented originals.\\n'
       'No new random image-level split allowed. Apply any ADDITIONAL augmentation only to train.\\n'
       'Do not claim independent natural broken-stitch test accuracy without new photos.\\n'
    )
    print('DATASET_CLEANED',json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--zip',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();main(a.zip,a.manifest,a.out)
