import numpy as np
import pandas as pd
from PIL import Image
import pytest
from broken_stitch.data import create_group_split
from broken_stitch.engine import train,evaluate,evaluate_ensemble


def build_manifest(tmp_path):
    rows=[]
    for source in range(20):
        for label in ('actual','faulty'):
            folder=tmp_path/'images'/label;folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{source:03}_{label}.png'
            a=np.full((28,28,3),200 if label=='actual' else 50,dtype=np.uint8)
            a[source%28,:]=(10+source)
            Image.fromarray(a).save(path)
            rows.append({'path':str(path),'label':label,'source_id':f'source_{source:03}','is_synthetic':'yes' if label=='faulty' else 'no'})
    raw=tmp_path/'raw.csv';pd.DataFrame(rows).to_csv(raw,index=False)
    out=tmp_path/'split.csv';create_group_split(raw,out)
    return out


def test_full_train_eval_and_ensemble(tmp_path):
    manifest=build_manifest(tmp_path)
    results=[]
    for seed in (1,2):
        out=tmp_path/f'run{seed}'
        r=train(str(manifest),out,model_name='cnn',epochs=1,patience=1,batch_size=8,image_size=32,seed=seed)
        assert r['model_weights_provenance_ok'] and not r['publication_ready']
        metrics=evaluate(r['checkpoint'],str(manifest),part='test')
        assert metrics['metrics']['n']>0
        results.append(r['checkpoint'])
    ens=evaluate_ensemble(results,str(manifest),part='test')
    assert ens['metrics']['n']>0


def test_manifest_tampering_detected(tmp_path):
    manifest=build_manifest(tmp_path)
    ckpt=train(str(manifest),tmp_path/'run',epochs=1,patience=1,batch_size=8,image_size=32)['checkpoint']
    df=pd.read_csv(manifest)
    # Rename source ID consistently, preserving group-exclusivity but invalidating training fingerprint.
    df.loc[df.source_id==df.source_id.iloc[0],'source_id']='new-source-id'
    df.to_csv(manifest,index=False)
    with pytest.raises(ValueError,match='Manifest differs'):
        evaluate(ckpt,str(manifest),part='test')