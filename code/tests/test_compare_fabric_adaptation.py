from pathlib import Path
import sys
import tempfile,csv
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_fabric_adaptation as m


def fake_oof(tmp):
    entries=[]
    # 11 fabric categories, full official binary subset size.
    for fabi,fab in enumerate(sorted(m.FABRICS)):
        n0=221 if fabi<10 else 226
        n1=109 if fabi<10 else 111
        for label,n in [('normal',n0),('broken',n1)]:
            for j in range(n):
                entries.append((f'source/{fab}_{label}_{j}.jpg',fab,label,fabi%5,
                                .85 if label=='broken' else .15))
    assert len(entries)==3637
    names={}
    for arm in m.ARMS:
        names[arm]=[]
        for k in range(5):
            p=tmp/f'{arm}-{k}.csv'
            with open(p,'w',newline='') as f:
                w=csv.writer(f);w.writerow(['source_file','fabric','actual_class','fold','probability_broken'])
                w.writerows(row for row in entries if row[3]==k)
            names[arm].append(p)
    return names


def test_fabric_comparison_accepts_matched_complete_inputs(tmp_path):
    p=fake_oof(tmp_path)
    result=m.compare(p,tmp_path/'result')
    assert result['n_images']==3637 and result['n_fabrics']==11
    assert result['models']['synthetic']['macro_f1']==1.0
    assert result['paired_11fabric_uncertainty']['synthetic_minus_imagenet_macro_f1']==0


def test_reject_any_fold_change_of_fabric_and_class(tmp_path):
    p=fake_oof(tmp_path)
    f=p['synthetic'][0]
    text=f.read_text()
    f.write_text(text.replace(',normal,0,',',broken,0,',1))
    with pytest.raises(ValueError,match='class counts wrong|unmatched label'):m.compare(p,tmp_path/'result')


def test_folds_require_all_11_fabrics_and_full_dataset(tmp_path):
    p=fake_oof(tmp_path)
    f=p['imagenet'][1]
    lines=f.read_text().splitlines();f.write_text('\n'.join(lines[:-1]))
    with pytest.raises(ValueError,match='all 3637'):m.read_oof(p['imagenet'])