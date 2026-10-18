from pathlib import Path
import pytest
from PIL import Image
from broken_stitch.external import make_external_manifest


def _im(p,color):
    p.parent.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(32,32),color).save(p)


def test_only_explicit_binary_directories(tmp_path):
    n=tmp_path/'normal';b=tmp_path/'broken';other=tmp_path/'crooked'
    _im(n/'n.png','white');_im(b/'b.png','black');_im(other/'other.png','red')
    summary=make_external_manifest(n,b,tmp_path/'manifest.csv')
    assert summary['n_images']==2
    assert summary['class_counts']=={'actual':1,'faulty':1}
    assert 'other' not in (tmp_path/'manifest.csv').read_text()


def test_incorrect_nested_class_roots_rejected(tmp_path):
    n=tmp_path/'normal';b=n/'broken'
    _im(n/'n.png','white');_im(b/'b.png','black')
    with pytest.raises(ValueError, match='must not contain'):
        make_external_manifest(n,b,tmp_path/'manifest.csv')


def test_identical_across_classes_rejected(tmp_path):
    n=tmp_path/'normal';b=tmp_path/'broken'
    _im(n/'same.png','white');_im(b/'same.png','white')
    with pytest.raises(ValueError,match='byte-identical'):
        make_external_manifest(n,b,tmp_path/'manifest.csv')


def test_external_evaluate_cnn_smoke(tmp_path):
    # Computer-only integration test; performance on toy pixels is meaningless.
    from test_training import build_manifest
    from broken_stitch.engine import train
    from broken_stitch.external import evaluate_external
    internal=build_manifest(tmp_path/'internal')
    ckpt=train(str(internal),tmp_path/'run',model_name='cnn',image_size=32,
               epochs=1,patience=1,batch_size=4,seed=5)['checkpoint']
    normal,broken=tmp_path/'external_normal',tmp_path/'external_broken'
    _im(normal/'a.png',(210,220,200));_im(broken/'b.png',(20,35,50))
    ext_csv=tmp_path/'external.csv'
    make_external_manifest(normal,broken,ext_csv)
    output=tmp_path/'result.json'
    result=evaluate_external(ckpt,internal,ext_csv,output)
    assert result['metrics']['n']==2
    assert result['publication_ready'] is False
    assert output.exists()