#!/usr/bin/env python3
"""Reproduce Fig. 1 using real archived stitch samples and a computed Sobel view.

Source archive figures page33_image1.jpeg and page33_image2.jpeg contain
eight-example contact sheets. These are NOT StitchingNet samples; the defects
in page33_image2 were edited. No model prediction is inferred or invented.
"""
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
import hashlib, math, numpy as np

SRC = Path(__file__).resolve().parent / "figures"
OUT = Path(__file__).resolve().parent / "figures_v15"
OUT.mkdir(exist_ok=True)
FILES = ["page33_image1.jpeg", "page33_image2.jpeg"]
samples=[]
for filename in FILES:
    fp=SRC/filename
    raw=fp.read_bytes()
    im=Image.open(fp).convert("RGB")
    assert im.width>1000 and im.height>650, (filename, im.size)
    print("SOURCE",filename,im.size,"sha256",hashlib.sha256(raw).hexdigest())
    # Crop the top-left actual stitch photograph from the 4x2 original contact sheet.
    # Coordinates are in original source-image pixels; never generate fake thread pixels.
    crop=im.crop((65,52,350,342))
    samples.append(crop)
normal, broken=samples
model_in=ImageOps.fit(normal,(224,224),method=Image.Resampling.BILINEAR)
gray=np.asarray(ImageOps.grayscale(model_in),dtype=np.float32)
# Explicit 3x3 Sobel magnitude over the exact resized sample.
p=np.pad(gray,1,mode="edge")
gx=(-p[:-2,:-2]+p[:-2,2:]-2*p[1:-1,:-2]+2*p[1:-1,2:]-p[2:,:-2]+p[2:,2:])
gy=(-p[:-2,:-2]-2*p[:-2,1:-1]-p[:-2,2:]+p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:])
mag=np.hypot(gx,gy)
mag=(255*mag/(mag.max()+1e-8)).clip(0,255).astype(np.uint8)
edge=Image.fromarray(mag,"L").convert("RGB")
# Four actual data-processing stages, no claim that this figure is a prediction.
W=1100; margin=26; gap=22; pane=(W-2*margin-3*gap)//4
panel_h=235; banner_h=35
canvas=Image.new("RGB",(W,banner_h+panel_h+30),"white")
d=ImageDraw.Draw(canvas)
try:
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15)
except OSError: font=ImageFont.load_default()
panels=[("(a) Correct sample",normal),("(b) Edited broken",broken),("(c) Resized 224x224",model_in),("(d) Sobel edges",edge)]
for j,(lab,src) in enumerate(panels):
    x=margin+j*(pane+gap)
    img=ImageOps.contain(src,(pane,panel_h),Image.Resampling.LANCZOS)
    y=banner_h+(panel_h-img.height)//2
    canvas.paste(img,(x+(pane-img.width)//2,y))
    d.rectangle((x,banner_h,x+pane,banner_h+panel_h),outline=(100,100,100),width=1)
    d.text((x+1,7),lab,fill=(15,15,15),font=font)
p=OUT/"original_stitch_preprocessing.png"
canvas.save(p,optimize=True)
print("OUTPUT",p,p.stat().st_size,"sha256",hashlib.sha256(p.read_bytes()).hexdigest())
