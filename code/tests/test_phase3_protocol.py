"""No ImageNet downloads or original private images in unit tests."""
import importlib.util,io,csv,uuid,zipfile
from pathlib import Path
import pytest

BASE=Path(__file__).resolve().parents[1]/'scripts'

def module(name):
    spec=importlib.util.spec_from_file_location(name,BASE/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def test_frozen_internal_manifest_keeps_parents_exclusive(tmp_path):
    p=tmp_path/'manifest.csv'
    fields=['relative_path','label','source_id','split']
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([
        dict(relative_path='a.jpg',label='actual',source_id='one',split='train'),
        dict(relative_path='b.jpg',label='faulty',source_id='one',split='train'),
        dict(relative_path='c.jpg',label='actual',source_id='two',split='val'),
        dict(relative_path='d.jpg',label='faulty',source_id='three',split='test')])
    rows,sha=module('mobilenet_feature_baseline').read_manifest(str(p))
    assert len(rows)==4 and len(sha)==64
    with p.open('a') as f:f.write('e.jpg,actual,one,test\n')
    with pytest.raises(ValueError,match='overlap'):
        module('mobilenet_feature_baseline').read_manifest(str(p))

def test_external_only_classes_00_and_02(tmp_path):
    p=tmp_path/'external.zip'
    with zipfile.ZipFile(p,'w') as z:
        for i in range(2436):z.writestr(f'StitchingNet/A. Cotton-Poly/0. Normal/A_00_{i:03d}.jpg',b'x')
        for i in range(1201):z.writestr(f'StitchingNet/A. Cotton-Poly/2. Broken stitch/A_02_{i:03d}.jpg',b'y')
        for i in range(35):z.writestr(f'StitchingNet/A. Cotton-Poly/1. Skipped/A_01_{i:03d}.jpg',b'z')
    rows=module('frozen_stitchingnet_eval').external_list(str(p))
    assert len(rows)==3637
    assert sum(label==1 for _,label in rows)==1201

def test_filename_photo_parent_mapping_is_cross_label(tmp_path):
    # 31 actual source families + 151 faulty Photoshop candidate families.
    n=2734;f=4631
    normals=[f'{i:02d}' for i in range(31)]
    # index = letters strictly: AA ... BE
    normals=[chr(ord('a')+i//26)+chr(ord('a')+i%26) for i in range(31)]
    faulty=[(k,(i%5)+1) for i,k in enumerate(normals*5)][:151]
    assert len(faulty)==151
    p=tmp_path/'archive.zip'
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_STORED) as z:
        for i in range(n):
            k=normals[i%len(normals)]
            z.writestr(f'dataset/train/actual/actual_original_{k.upper()}.png_{uuid.uuid4()}.png',b'')
        for i in range(f):
            # ensure unique 151 fault families, build unique suffix per normal
            k=faulty[i%len(faulty)]
            z.writestr(f'dataset/val/faulty/faulty_original_{k[0].upper()}{k[1]}.png_{uuid.uuid4()}.png',b'')
    # generated (normal, photo-label) families are unique? the faulty list cycles duplicate pairs, only 31*5 =155 but first151 yes unique via index modulo.
    result=tmp_path/'result'
    module('photo_parent_filename_map').run(str(p),str(result))
    import json
    j=json.loads((result/'photo_parent_filename_mapping_summary.json').read_text())
    assert j['inferred_underlying_photo_count']==31 and j['faulty_photoshop_candidate_count']==151
    assert j['inferred_parent_match_rate']==1