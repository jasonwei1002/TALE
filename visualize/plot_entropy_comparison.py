"""Quantitative comparison of attention sharpness:
TALE (full) vs TALE w/o local loss.

For each test sample with at least one positive label, we form a CKEPE prompt
from its present labels, compute the GLoRIA-style word-to-patch attention with
both models, and aggregate per-word entropy:

    H_word = -∑_t p(t|word) · log p(t|word)

Lower entropy → more spatially focused attention → stronger local alignment.

We then:
  * average per-sample H across diagnostic words → one scalar per sample,
  * plot paired box/violin (full vs ablation) over all samples,
  * report a paired t-test (and Wilcoxon signed-rank as a non-parametric
    backup) p-value in the figure.

Usage (from repo root):
    python visualize/plot_entropy_comparison.py \
        --ckpt        checkpoints/best/vit_small_bestZeroShotAll_ckpt.pth \
        --ckpt-ablation checkpoints/去掉局部对比/vit_small_bestZeroShotAll_ckpt.pth \
        --dataset     ptbxl-form \
        --output      visualize/fig-attention-entropy.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "finetune"))

# Re-use heavy lifting from the qualitative script
from plot_attention_map import (  # noqa: E402
    DATASET_CONFIGS,
    extract_attention,
    fix_hyphenated_words,
    load_model,
    merge_subwords,
    select_top_words,
    apply_word_indices,
)
from finetune_dataset import ECGDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Per-sample attention entropy: full vs ablation")
    p.add_argument(
        "--ckpt",
        default=str(PROJECT_ROOT / "checkpoints/best/vit_small_bestZeroShotAll_ckpt.pth"),
    )
    p.add_argument(
        "--ckpt-ablation",
        required=True,
        help="w/o local loss checkpoint",
    )
    p.add_argument(
        "--data_root",
        default=str(PROJECT_ROOT.parent / "datasets" / "finetune"),
    )
    p.add_argument(
        "--dataset",
        default="ptbxl-form",
        choices=sorted(DATASET_CONFIGS.keys()),
    )
    p.add_argument(
        "--prompt_json",
        default=str(PROJECT_ROOT / "zeroshot" / "CKEPE_prompt.json"),
    )
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "visualize" / "fig-attention-entropy.png"),
    )
    p.add_argument("--max-samples", type=int, default=500,
                   help="cap test samples for speed (0 = use all)")
    p.add_argument("--max-words", type=int, default=8,
                   help="top-K diagnostic words per sample")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def attn_entropy(attn: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-row Shannon entropy. attn: [n_words, n_patches]; returns [n_words]."""
    p = attn / (attn.sum(axis=1, keepdims=True) + eps)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def per_sample_entropy(model, ecg, prompt_text, device, max_words):
    """Return mean per-word entropy of the diagnostic words for one sample."""
    attn, tokens = extract_attention(model, ecg, prompt_text, device)
    attn, tokens = merge_subwords(attn, tokens)
    attn, tokens = fix_hyphenated_words(attn, tokens)
    keep, dedup_labels = select_top_words(attn, tokens, max_words=max_words)
    attn_keep, _ = apply_word_indices(attn, tokens, keep, dedup_labels)
    return float(attn_entropy(attn_keep).mean())


def collect_paired_entropies(
    model_full, model_abl, dataset, prompt_dict, device, max_samples, max_words, seed
):
    rng = np.random.RandomState(seed)
    labels = dataset.labels
    label_names = dataset.labels_name

    # 只评估"至少有一个 positive label 且至少一个 label 有 prompt"的样本
    valid_indices = []
    for i in range(len(labels)):
        present = [label_names[j] for j in range(labels.shape[1]) if labels[i, j] == 1]
        present = [n for n in present if n in prompt_dict]
        if present:
            valid_indices.append(i)
    valid_indices = np.array(valid_indices)
    if max_samples and len(valid_indices) > max_samples:
        valid_indices = rng.choice(valid_indices, size=max_samples, replace=False)

    print(f"  evaluating on {len(valid_indices)} samples")

    full_h, abl_h = [], []
    for idx in tqdm(valid_indices, desc="entropy"):
        ecg, _ = dataset[int(idx)]
        present = [label_names[j] for j in range(labels.shape[1]) if labels[int(idx), j] == 1]
        present = [n for n in present if n in prompt_dict]
        prompt_text = " ".join(prompt_dict[n] for n in present)
        try:
            h_full = per_sample_entropy(model_full, ecg, prompt_text, device, max_words)
            h_abl = per_sample_entropy(model_abl, ecg, prompt_text, device, max_words)
        except Exception as e:
            print(f"  [skip idx={idx}] {e}")
            continue
        full_h.append(h_full)
        abl_h.append(h_abl)

    return np.asarray(full_h), np.asarray(abl_h)


def plot_paired(full_h, abl_h, output_path):
    """Paired-difference histogram with Cohen's d_z and improved%.

    For paired data, the only honest visualization is the distribution of
    differences: Δ = H_full − H_abl. Bars left of 0 → full has lower entropy
    (sharper alignment); bars right of 0 → ablation has lower entropy.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.6,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    full_h = np.asarray(full_h)
    abl_h = np.asarray(abl_h)
    n = len(full_h)
    delta = full_h - abl_h           # 负值 = 完整模型熵更小（更尖锐）

    # ---- 统计量 ----
    t_stat, p_t = stats.ttest_rel(full_h, abl_h)
    w_stat, p_w = stats.wilcoxon(full_h, abl_h)
    mean_d = float(delta.mean())
    median_d = float(np.median(delta))
    sd_d = float(delta.std(ddof=1))
    cohen_dz = mean_d / sd_d if sd_d > 0 else float("nan")
    se = sd_d / np.sqrt(n)
    ci_low, ci_high = mean_d - 1.96 * se, mean_d + 1.96 * se
    n_improved = int((delta < 0).sum())
    pct_improved = 100.0 * n_improved / n

    print(f"\n--- paired-difference statistics ---")
    print(f"  n            = {n}")
    print(f"  Δ̄            = {mean_d:+.4f}   (95% CI [{ci_low:+.4f}, {ci_high:+.4f}])")
    print(f"  median(Δ)    = {median_d:+.4f}")
    print(f"  SD(Δ)        = {sd_d:.4f}")
    print(f"  Cohen's dz   = {cohen_dz:+.3f}")
    print(f"  improved%    = {pct_improved:.1f}% ({n_improved}/{n} samples)")
    print(f"  paired t     = {t_stat:.2f},  p = {p_t:.2e}")
    print(f"  Wilcoxon     = {w_stat:.1f}, p = {p_w:.2e}")

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(3.6, 2.9))

    n_bins = 35
    color_neg = "#a50f15"   # red 浪漫色：full 更优
    color_pos = "#bdbdbd"   # gray：full 更差
    bin_edges = np.linspace(delta.min(), delta.max(), n_bins + 1)
    counts, _ = np.histogram(delta, bins=bin_edges)
    for i, c in enumerate(counts):
        mid = 0.5 * (bin_edges[i] + bin_edges[i + 1])
        bar_color = color_neg if mid < 0 else color_pos
        ax.bar(
            bin_edges[i], c, width=bin_edges[i + 1] - bin_edges[i],
            align="edge", color=bar_color,
            edgecolor="black", linewidth=0.3, alpha=0.85,
        )

    # 0 参考线
    ymax = ax.get_ylim()[1]
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, zorder=4)
    ax.text(0, ymax * 0.97, " no effect", fontsize=7,
            va="top", ha="left", color="black")

    # 均值 Δ̄
    ax.axvline(mean_d, color="#0b3d91", linestyle="-", linewidth=1.4, zorder=5)
    ax.text(mean_d, ymax * 0.83,
            f" $\\bar{{\\Delta}}={mean_d:+.3f}$",
            fontsize=8, va="top", ha="left", color="#0b3d91")

    ax.set_xlabel(r"$\Delta$ entropy = $H_{\mathrm{full}} - H_{\mathrm{w/o\ local}}$",
                  fontsize=9)
    ax.set_ylabel("# samples", fontsize=9)

    p_exp = int(np.floor(np.log10(max(p_t, 1e-300))))
    title = (
        "TALE (full) reduces attention entropy\n"
        f"$d_z={cohen_dz:.2f}$, "
        f"{pct_improved:.0f}% improved, "
        f"$p<10^{{{p_exp}}}$  ($n={n}$)"
    )
    ax.set_title(title, fontsize=8.5, pad=5)

    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n图已保存: {output_path}")
    plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = DATASET_CONFIGS[args.dataset]
    print(f"[1/4] loading dataset: {args.dataset}")
    dataset = ECGDataset(
        data_path=os.path.join(args.data_root, cfg["data_subdir"]),
        csv_file=os.path.join(args.data_root, cfg["split_subpath"]),
        mode="test",
        dataset_name=cfg["dataset_name"],
    )
    print(f"  size={len(dataset)} labels={dataset.labels_name}")

    print(f"\n[2/4] loading models")
    print(f"  full: {args.ckpt}")
    model_full = load_model(args.ckpt, device)
    print(f"  abl : {args.ckpt_ablation}")
    model_abl = load_model(args.ckpt_ablation, device)

    print(f"\n[3/4] loading prompts: {args.prompt_json}")
    with open(args.prompt_json) as f:
        prompt_dict = yaml.load(f, Loader=yaml.FullLoader)

    print(f"\n[4/4] computing per-sample attention entropies")
    full_h, abl_h = collect_paired_entropies(
        model_full, model_abl, dataset, prompt_dict, device,
        max_samples=(0 if args.max_samples == 0 else args.max_samples),
        max_words=args.max_words, seed=args.seed,
    )

    plot_paired(full_h, abl_h, args.output)


if __name__ == "__main__":
    main()
