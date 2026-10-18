"""Candidate cross-class parent links, NOT ground-truth photo lineage. GitHub CI artifact.
Produces ranking and contact sheets for HUMAN photo-source verification.
"""
import argparse,csv,hashlib,json,re,zipfile,io,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
import cv2

UUID=re.compile(r'^(?P<stem>.+\.(?:png|jpe?g))_[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\.(?:png|jpe?g)$',re.I)

def classify(member):
    x=Path(member); cls={v.lower() for v in x.parts} & {'actual','faulty'}
    m=UUID.fullmatch(x.name)
    if len(cls)!=1 or not m:return None
    return next(iter(cls)), m.group('stem').lower()

def getimg(z,name):
    with z.open(name) as f:
        data=f.read()
    with Image.open(io.BytesIO(data)) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((320,320))
        return im.copy()

def features(im):
    arr=np.array(im.resize((112,112),Image.Resampling.BILINEAR))
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    # reject brittle exact geometry: color histogram and Fourier low frequencies
    hist=[]
    for c in range(3):
        h=cv2.calcHist([arr],[c],None,[24],[0,256]).ravel()
        hist.extend((h/(np.linalg.norm(h)+1e-8)).tolist())
    # 32x32 luminosity after contrast normalization, shift tolerance from embedding is limited.
    small=cv2.resize(gray,(32,32)).astype('float32')
    norm=(small-small.mean())/(small.std()+1e-5)
    dct=cv2.dct(norm)[:8,:8].ravel().astype('float32')
    dct=dct/(np.linalg.norm(dct)+1e-8)
    return np.asarray(hist,'float32'),dct,gray

def sim(a,b):
    histA,dctA,imgA=a;histB,dctB,imgB=b
    hist=float(np.dot(histA,histB)/3)
    cosine=float(np.dot(dctA,dctB))
    # ORB invariant correspondence indicates same photographed textured region, not just same color
    o=cv2.ORB_create(nfeatures=300,scaleFactor=1.2)
    ka,da=o.detectAndCompute(imgA,None);kb,db=o.detectAndCompute(imgB,None)
    orb=0.0;matches=0
    if da is not None and db is not None and len(da)>3 and len(db)>3:
        ms=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(da,db)
        ms=sorted(ms,key=lambda k:k.distance)
        matches=sum(x.distance<45 for x in ms)
        orb=min(matches,25)/25
    # score is ranking only, not match probability
    return .32*hist+.30*(cosine+1)/2+.38*orb,{'hist':round(hist,4),'dct':round(cosine,4),'orb_good':matches}

def main(zpath,outpath):
    out=Path(outpath);out.mkdir(parents=True,exist_ok=True)
    group=defaultdict(list)
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            cl=classify(n)
            if cl:group[cl].append(n)
        groups={}
        for (lab,stem),names in sorted(group.items()):
            names.sort()
            # first, median, last augmented image to allow rotation/random transformation
            idx=sorted({0,len(names)//2,len(names)-1})
            chosen=[]
            for i in idx:
                try:
                    im=getimg(z,names[i]);chosen.append((names[i],im,features(im)))
                except (OSError,ValueError):pass
            if chosen:groups[(lab,stem)]=chosen
    normals=sorted(k for k in groups if k[0]=='actual')
    faults=sorted(k for k in groups if k[0]=='faulty')
    allr=[]
    for nk in normals:
        for fk in faults:
            results=[(sim(a[2],b[2]),a[0],b[0],a[1],b[1]) for a in groups[nk] for b in groups[fk]]
            (score,comp),nameA,nameB,imgA,imgB=max(results,key=lambda v:v[0][0])
            allr.append({'normal_candidate_family':nk[1],'faulty_candidate_family':fk[1],
                          'score_rank_only':round(score,6),**comp,'normal_filename':nameA,'faulty_filename':nameB})
    allr.sort(key=lambda x:x['score_rank_only'],reverse=True)
    with (out/'candidate_cross_class_pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allr[0]));w.writeheader();w.writerows(allr)
    # strongest candidates: 5 fault families per one normal family; do not infer verified links
    bynormal=defaultdict(list)
    for row in allr:
        if len(bynormal[row['normal_candidate_family']])<5:
            bynormal[row['normal_candidate_family']].append(row)
    (out/'review_candidates_top5.json').write_text(json.dumps(dict(bynormal),indent=2))
    # 3 pages: 12 original normal families per contact sheet; five closest faulty-family matches.
    with zipfile.ZipFile(zpath) as z:
        for part in range(math.ceil(len(normals)/12)):
            selected=normals[part*12:(part+1)*12]
            cw,ch=145,170
            canvas=Image.new('RGB',(cw*6,ch*len(selected)),(245,245,245))
            d=ImageDraw.Draw(canvas)
            for row_idx,nk in enumerate(selected):
                ranks=bynormal[nk[1]]
                im=groups[nk][0][1].resize((138,138))
                canvas.paste(im,(0,row_idx*ch+20))
                d.text((3,row_idx*ch+2),'NORMAL '+nk[1][:15],fill=(10,10,10))
                for j,match in enumerate(ranks):
                    # display the actual chosen pair, not a generic image from the family
                    im=getimg(z,match['faulty_filename']).resize((138,138))
                    canvas.paste(im,((j+1)*cw,row_idx*ch+20))
                    d.text(((j+1)*cw,row_idx*ch+2),f'F{j+1} score={match["score_rank_only"]:.2f}',fill=(25,25,25))
            canvas.save(out/f'lineage_contact_sheet_{part+1}.jpg',quality=82)
    summary={'status':'AUTOMATED_RANKING_ONLY_NOT_PHOTO_LINEAGE_VERIFICATION',
             'num_normal_candidate_groups':len(normals),'num_faulty_candidate_groups':len(faults),
             'num_candidate_pairs_compared':len(allr),'files':sorted(x.name for x in out.iterdir()),
             'limitations':['Photoshop source photo identity is not proven by similarity scores',
                            'Image augmentations and backgrounds may inflate similarity',
                            'Only three representatives per candidate family used',
                            'Manual expert inspection + raw photo provenance needed to confirm mappings'],
             'top_global_pairs':allr[:12]}
    (out/'lineage_review_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='top_global_pairs'},indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--zip',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args();main(a.zip,a.out)