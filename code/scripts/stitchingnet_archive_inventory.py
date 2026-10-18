"""Audit officially published StitchingNet ZIP metadata; do NOT evaluate or tune models.
Official normal code 00 and broken-stitch 02 (all other defect codes excluded).
"""
import argparse,zipfile,json,re,collections,hashlib
from pathlib import Path
RE=re.compile(r'^(?P<fabric>[A-K])_(?P<code>\d{2})_(?P<sample>\d+)\.(?:jpg|jpeg|png)$',re.I)

def main(archive,out):
    sha=hashlib.md5()
    with open(archive,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):sha.update(chunk)
    if sha.hexdigest() != '51b228d9572b925035e9e9af01c96569':
        raise ValueError('STOP: unexpected StitchingNet archive checksum: '+sha.hexdigest())
    with zipfile.ZipFile(archive) as z:
        allnames=[x.filename for x in z.infolist() if not x.is_dir()]
        matches=[];ignored=[]
        for name in allnames:
            n=RE.fullmatch(Path(name).name)
            if n and name.lower().endswith(('.jpg','.jpeg','.png')):
                matches.append({'entry':name,'class':n['code'],'fabric':n['fabric'],'sample':n['sample']})
            else:ignored.append(name)
        ct=collections.Counter(x['class'] for x in matches)
        fabrics={k:dict(collections.Counter(x['fabric'] for x in matches if x['class']==k)) for k in ('00','02')}
    # Only assert publisher-supported count; otherwise stop and ask for manual adaptation
    if ct['00']!=2436 or ct['02']!=1201 or len(matches)!=14565:
        raise ValueError(f'STOP: Unexpected official dataset layout/counts: {ct}; examples={matches[:10]} ignored={ignored[:10]}')
    report={'archive_md5':sha.hexdigest(),'n_archive_members':len(allnames),'n_labelled_images':len(matches),
            'class_codes_counts':dict(ct),'binary_subset_classes':{'00':'actual','02':'faulty'},
            'n_binary':ct['00']+ct['02'],'fabric_counts_binary':fabrics,
            'sample_paths':matches[:15],'ignored_entries_sample':ignored[:15],
            'source':'https://zenodo.org/records/20352813','license':'CC BY-NC 4.0 per official StitchingNet GitHub',
            'evaluation_performed':False}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('sample_paths','ignored_entries_sample')},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.zip,a.out)