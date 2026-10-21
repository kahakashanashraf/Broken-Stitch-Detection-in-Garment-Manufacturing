"""Reproduce the two embedded v13 paper figures from verified experiment evidence.

NOT a simulation of CAM. Downloads are handled by the calling workflow from
thesis-only-vs-fabric-heldout-gradcam-v1: two original, released paired PNGs.
Each grayscale intensity is converted through the same fixed monotonic colormap.
No input photograph pixels are copied into the paper figure.
"""
from pathlib import Path
import argparse
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

EXPECTED = {
    "fold0_true_positive_paired_gradcam.png": "f5781ee133d98c4900ea4b6a6d8475b4354d78334b3a57c79acc7b3adeec930e",
    "fold2_false_negative_paired_gradcam.png": "c52da75691e9fb2cd80f7a88c6a4ca40bcafac9edf6144ae90d964c752053ffc",
}


def make_figures(xai_dir, out_dir):
    xai_dir, out_dir = Path(xai_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for fn, sha in EXPECTED.items():
        raw = (xai_dir / fn).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("STOP: original Grad-CAM asset checksum differs: " + fn)
        im = Image.open(xai_dir / fn).convert("RGB")
        if im.size != (448, 260):
            raise ValueError("STOP: original Grad-CAM panel dimensions changed")
        sources.append(im)

    # The original evidence PNGs contain two adjacent 224x224 maps and
    # a ~36px header. Crop headers only; leave all saliency pixels untouched.
    regions = [
        sources[0].crop((0, 36, 224, 260)),
        sources[0].crop((224, 36, 448, 260)),
        sources[1].crop((0, 36, 224, 260)),
        sources[1].crop((224, 36, 448, 260)),
    ]
    labels = [
        ("Fabric D / edited-only", "p(broken) = 0.053"),
        ("Fabric D / fabric-trained", "p(broken) = 0.664"),
        ("Fabric B / edited-only", "p(broken) = 0.988"),
        ("Fabric B / fabric-trained", "p(broken) = 0.259"),
    ]
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    f_label = ImageFont.truetype(bold, 20)
    f_secondary = ImageFont.truetype(font, 19)
    canvas = Image.new("RGB", (1050, 1050), "white")
    draw = ImageDraw.Draw(canvas)
    for j, src in enumerate(regions):
        # The two earlier model runs normalize their CAMs independently.
        # Do not renormalize to equalize maxima or generate new hot spots.
        arr = np.asarray(src.convert("L"), dtype=np.float32) / 255.0
        heat = np.uint8(np.round(colormaps["inferno"](np.clip(arr, 0, 1))[:, :, :3] * 255))
        color = Image.fromarray(heat).resize((482, 388), Image.Resampling.LANCZOS)
        x, y = (23, 545)[j % 2], (48, 555)[j // 2]
        draw.text((x, y - 35), labels[j][0], font=f_label, fill="#182531")
        draw.text((x, y - 8), labels[j][1], font=f_secondary, fill="#404b54")
        canvas.paste(color, (x, y + 28))
        draw.rectangle((x, y + 28, x + 481, y + 415), outline="#838a92", width=2)
    draw.text((24, 1020), "Published Grad-CAM only; coarse 7 x 7 map before resizing.",
              font=f_secondary, fill="#414d55")
    gradcam = out_dir / "verified_cam_colorized_D_B.png"
    canvas.save(gradcam, optimize=True)

    groups = ["D/E", "I/J", "B/F", "C/G/K", "A/H"]
    imagenet = [0.75384, 0.68665, 0.96959, 0.71661, 0.90538]
    edited = [0.78064, 0.69312, 0.96341, 0.73857, 0.91257]
    fig, ax = plt.subplots(figsize=(6.35, 3.25), dpi=220)
    xx, w = np.arange(5), .34
    ax.bar(xx - w / 2, imagenet, w, label="ImageNet init.", color="#7399b8")
    ax.bar(xx + w / 2, edited, w, label="Edited-image init.", color="#cf9260")
    ax.set_xticks(xx, groups)
    ax.set_xlabel("Held-out fabric groups")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout(pad=.4)
    plot = out_dir / "heldout_fabric_macro_f1.png"
    fig.savefig(plot, bbox_inches="tight")
    plt.close(fig)
    print("Verified figure outputs:", gradcam, plot)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xai-dir", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    make_figures(a.xai_dir, a.out_dir)
