"""Read-only inspection of original thesis dataset ZIP; does not assign source IDs.

Outputs contain aggregate information and a sample of paths, not the actual images.
Never treat class names or filename stems as verified real-world provenance.
"""
import argparse
import collections
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from PIL import Image, ImageOps

EXT={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
GOOD={'actual','normal','good','correct','ok','healthy'}
BAD={'faulty','broken','defective','defect','wrong','bad'}


def audit(zippath,reportpath,samples=80):
    zp=Path(zippath)
    row=[]; path_leaves=collections.Counter(); directories=collections.Counter(); roots=collections.Counter(); ext=collections.Counter()
    sizes=collections.Counter(); widths=collections.Counter(); guessed=collections.Counter(); statuses=collections.Counter()
    hashes=collections.defaultdict(list); pixel_hashes=collections.defaultdict(list); small=[]
    with zipfile.ZipFile(zp) as z:
        bad_member=z.testzip()
        if bad_member: raise ValueError(f'ZIP CRC failure: {bad_member}')
        members=z.infolist()
        fileinfos=[x for x in members if not x.is_dir()]
        images=[x for x in fileinfos if PurePosixPath(x.filename).suffix.lower() in EXT]
        for info in fileinfos:
            p=PurePosixPath(info.filename.replace('\\','/'))
            roots[p.parts[0] if p.parts else 'EMPTY']+=1
            ext[p.suffix.lower()]+=1
        for info in images:
            p=PurePosixPath(info.filename.replace('\\','/'))
            path_leaves[str(p.parent)]+=1
            for i in range(1,min(4,len(p.parts))):directories['/'.join(p.parts[:i])]+=1
            tokens=[t.lower() for t in p.parts[:-1]]
            stem=p.stem.lower()
            labels={'normal' if t in GOOD else 'faulty' for t in tokens if t in GOOD|BAD}
            if len(labels)==1:label=next(iter(labels))
            elif len(labels)>1:label='ambiguous'
            else:label='unknown'
            guessed[label]+=1
            entry={'path':info.filename,'size':info.file_size,'folder':str(p.parent),'label_from_folder':label,'filename_stem':p.stem}
            try:
                data=z.read(info)
                sha=hashlib.sha256(data).hexdigest()
                hashes[sha].append(info.filename)
                with Image.open(io.BytesIO(data)) as im:
                    im=ImageOps.exif_transpose(im)
                    w,h=im.size;im.load()
                    widths[f'{w}x{h}']+=1
                    # hash of grayscale 16x16 thumbnail only, exploratory near-duplicate warning
                    thumb=im.convert('L').resize((16,16),Image.Resampling.BILINEAR).tobytes()
                    ph=hashlib.sha256(thumb).hexdigest()
                    pixel_hashes[ph].append(info.filename)
                statuses['decoded_ok']+=1
                entry['dimensions']=f'{w}x{h}'
            except Exception as e:
                statuses['decode_error']+=1;entry['error']=type(e).__name__+': '+str(e)[:100]
            sizes[label]+=info.file_size
            row.append(entry)
        dupe=[v for v in hashes.values() if len(v)>1]
        near=[v for v in pixel_hashes.values() if len(v)>1]
        report={
            'archive':zp.name,'archive_size_bytes':zp.stat().st_size,
            'zip_integrity':'PASS','archive_total_members':len(members),'archive_files':len(fileinfos),
            'images':len(images),'other_files':len(fileinfos)-len(images),
            'root_prefix_counts':roots.most_common(25),
            'image_folder_counts':path_leaves.most_common(150),
            'image_class_from_folder_counts':dict(guessed),
            'extension_counts':ext.most_common(25),
            'image_dimensions':widths.most_common(20),
            'image_bytes_by_folder_guessed_label':dict(sizes),
            'decode_status':dict(statuses),
            'exact_duplicate_image_hash_groups':len(dupe),
            'exact_duplicate_image_instances':sum(len(g)-1 for g in dupe),
            'exact_duplicate_image_examples':[g[:6] for g in dupe[:15]],
            'same_low_res_thumbnail_hash_groups':len(near),
            'same_low_res_thumbnail_hash_examples':[g[:5] for g in near[:15]],
            'first_image_filenames': [x['path'] for x in row[:samples]],
            'warning':'Filename/label/synthetic lineage are NOT verified; do NOT infer independent source_id from image filenames without provenance audit.',
            'training_gate':'BLOCKED until original-image to augmentation/Photoshop parentage proven'
        }
        Path(reportpath).parent.mkdir(parents=True,exist_ok=True)
        Path(reportpath).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('first_image_filenames','image_folder_counts','exact_duplicate_image_examples','same_low_res_thumbnail_hash_examples')},indent=2,ensure_ascii=False),flush=True)
        print('Top image folders:',path_leaves.most_common(30),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--zip',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    audit(args.zip,args.out)