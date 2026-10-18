"""Generate visualization using ONLY results from this reconstruction's checkpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_from_files(history_json,eval_json,outdir):
    h=json.loads(Path(history_json).read_text(encoding='utf8'))
    e=json.loads(Path(eval_json).read_text(encoding='utf8'))
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    if not h:raise ValueError('Empty history')
    fig,ax=plt.subplots(figsize=(6,3.5))
    ax.plot([x['epoch'] for x in h],[x['val_macro_f1'] for x in h],label='validation macro-F1')
    ax.plot([x['epoch'] for x in h],[x['val_balanced_accuracy'] for x in h],label='validation balanced accuracy')
    ax.set(xlabel='Epoch',ylabel='Score',ylim=(0,1),title='Reconstructed experiment (not 2022 thesis result)')
    ax.legend();fig.tight_layout();fig.savefig(out/'validation_curves.png',dpi=200);plt.close(fig)
    cm=np.asarray(e['metrics']['confusion_matrix_actual_faulty'])
    fig,ax=plt.subplots(figsize=(4,4));im=ax.imshow(cm,cmap='Blues')
    for (i,j),value in np.ndenumerate(cm):ax.text(j,i,str(value),ha='center',va='center')
    ax.set_xticks([0,1],labels=['actual','faulty']);ax.set_yticks([0,1],labels=['actual','faulty'])
    ax.set(xlabel='Predicted',ylabel='True',title=f"Confusion matrix — {e['split']}")
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=200);plt.close(fig)
    return [str(out/'validation_curves.png'),str(out/'confusion_matrix.png')]