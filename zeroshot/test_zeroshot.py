import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append("../utils")
import utils_builder
from zeroshot_val import zeroshot_eval, get_zero_dataset
from config import config

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device_id = 'cuda'

torch.manual_seed(42)
random.seed(0)
np.random.seed(0)

model = utils_builder.ECGCLIP(config['network'])
ckpt = '/public/home/hs_mmcd_5/project/jasonwei/MERL/checkpoints/best/resnet18_bestZeroShotAll_ckpt.pth'
ckpt = torch.load(f'{ckpt}', map_location='cpu')
model.load_state_dict(ckpt)
model = model.to(device_id)
model = torch.nn.DataParallel(model)

args_zeroshot_eval = config['zeroshot']

avg_f1, avg_acc, avg_auc = 0, 0, 0
for set_name in args_zeroshot_eval['test_sets'].keys():

        f1, acc, auc, _, _, _, res_dict = \
        zeroshot_eval(model=model,
        set_name=set_name,
        device=device_id,
        args_zeroshot_eval=args_zeroshot_eval)

        avg_f1 += f1
        avg_acc += acc
        avg_auc += auc

avg_f1 = avg_f1/len(args_zeroshot_eval['test_sets'].keys())
avg_acc = avg_acc/len(args_zeroshot_eval['test_sets'].keys())
avg_auc = avg_auc/len(args_zeroshot_eval['test_sets'].keys())

print(f'avg_f1: {avg_f1}, avg_acc: {avg_acc}, avg_auc: {avg_auc}')

# ── t-SNE visualization on Chapman (CSN) test set ────────────────────────────

def extract_ecg_embeddings(model, loader, device: str) -> np.ndarray:
    """Return L2-normalized ECG embeddings, shape (N, D)."""
    model.eval()
    embs = []
    with torch.no_grad():
        for ecg, _ in tqdm(loader, desc="Extracting embeddings"):
            ecg = ecg.to(device)
            emb = model.ext_ecg_emb(ecg)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray,
              class_names: list, save_path: str, title: str = "TALE") -> None:
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40,
                max_iter=1000, init="pca")
    coords = tsne.fit_transform(embeddings)

    # Target classes to display (abbreviation → color)
    TARGET = {
        "SA":  "#FF0000",
        "ALS": "#00AA00",
        "APB": "#0000FF",
        "AF":  "#00CCCC",
        "SVT": "#CC00CC",
        "TWC": "#AAAA00",
        "ST":  "#000000",
    }
    TARGET_ORDER = ["SA", "ALS", "APB", "AF", "SVT", "TWC", "ST"]

    # Map full class names to abbreviations using suffix matching
    def to_abbrev(name: str) -> str:
        name_l = name.lower()
        mapping = {
            "sinus arrhythmia": "SA",
            "atrial left shift": "ALS", "left shift": "ALS",
            "atrial premature": "APB", "premature atrial": "APB",
            "atrial fibrillation": "AF",
            "supraventricular tachycardia": "SVT",
            "t-wave change": "TWC", "t wave change": "TWC", "two": "TWC",
            "st-change": "ST", "st change": "ST", "stdd": "ST",
        }
        for key, abbr in mapping.items():
            if key in name_l:
                return abbr
        return name  # keep original if no match

    abbrev_names = [to_abbrev(n) for n in class_names]
    primary_all = np.argmax(labels, axis=1)

    # Build index of which dataset class indices are in TARGET
    target_class_indices = [i for i, a in enumerate(abbrev_names) if a in TARGET]
    keep_mask = np.isin(primary_all, target_class_indices)

    print(f"Filtering: keeping {keep_mask.sum()} / {len(keep_mask)} samples "
          f"({', '.join(abbrev_names[i] for i in target_class_indices)})")

    embeddings_f = embeddings[keep_mask]
    # Map primary label → abbreviation string (merge multiple dataset classes → same abbrev)
    primary_abbrev_f = np.array([abbrev_names[i] for i in primary_all[keep_mask]])

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40,
                max_iter=1000, init="pca", early_exaggeration=4.0)
    coords = tsne.fit_transform(embeddings_f)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    ax.set_xticks([])
    ax.set_yticks([])

    for abbrev in TARGET_ORDER:
        mask = primary_abbrev_f == abbrev
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=TARGET[abbrev], s=25, alpha=0.75, linewidths=0,
            label=abbrev,
        )

    ax.set_title(title, fontweight="bold", pad=22)
    ax.legend(
        markerscale=2.0, fontsize=8, frameon=True,
        loc="upper center", ncol=len(TARGET_ORDER),
        bbox_to_anchor=(0.5, 1.18),
        handletextpad=0.3, columnspacing=0.6,
    )

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved t-SNE figure → {save_path}")


# Build Chapman test dataset (same path logic as zeroshot_eval)
import os as _os
_args = args_zeroshot_eval
_meta = _args['meta_data_path']
_split_root = _args['meta_split_path']
_chapman_cfg = _args['test_sets']['chapman']
_data_path = _os.path.join(_meta, _chapman_cfg['data_path'])
_split_path = _os.path.join(_split_root, _chapman_cfg['split_path'])

sys.path.append("../finetune")
from finetune_dataset import getdataset as get_zero_dataset

chapman_dataset = get_zero_dataset(_data_path, _split_path,
                                   mode='test', dataset_name='chapman')
chapman_loader = DataLoader(
    chapman_dataset,
    batch_size=256,
    num_workers=_args['num_workers'],
    pin_memory=True,
    shuffle=False,
    drop_last=False,
)

embeddings = extract_ecg_embeddings(model.module, chapman_loader, device_id)
labels = np.array(chapman_dataset.labels)
class_names = chapman_dataset.labels_name

save_path = _os.path.join(_os.path.dirname(__file__), "tsne_chapman.pdf")
plot_tsne(embeddings, labels, class_names, save_path, title="TALE")
