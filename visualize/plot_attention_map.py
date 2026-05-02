"""
局部对齐 Attention 矩阵可视化脚本

从 PTB-XL Super 测试集中选取多标签样例，
拼接各标签 CKEPE prompt 后提取 ECG patch--诊断词语 attention 矩阵，
绘制双栏并排热力图以展示不同词语对不同时间段的局部对应关系。

用法:
    python visualize/plot_attention_map.py \
        --ckpt /path/to/bestZeroShotAll_ckpt.pth \
        --data_root /path/to/datasets/finetune \
        --output figures/fig-attention-map.png
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- 路径设置 ----
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "finetune"))

from utils_builder import ECGCLIP
from finetune_dataset import ECGDataset


# ---- 超参数（与训练一致） ----
TEMP1 = 4.0   # attention 温度
TEMP2 = 5.0   # 相似度缩放
PROJ_DIM = 256

# 可视化参数：将连续 patch 聚合为 segment 以提高可读性
# 200 patches / 20 = 10 segments，每段对应 1.0s
SEGMENT_SIZE = 20


# 各 PTB-XL 子任务的可视化预设。
#   data_subdir:   ECG 文件根目录相对 data_root 的子路径
#   split_subpath: 测试 CSV 相对 data_root 的子路径
#   dataset_name:  传给 ECGDataset 的 dataset_name
#   default_combos: 默认两个最具戏剧性局部对齐效果的标签组合
DATASET_CONFIGS = {
    "ptbxl-super": {
        "data_subdir": "ptbxl",
        "split_subpath": "data_split/ptbxl/super_class/ptbxl_super_class_test.csv",
        "dataset_name": "ptbxl",
        "default_combos": [["MI", "STTC"], ["CD", "HYP"]],
    },
    "ptbxl-sub": {
        "data_subdir": "ptbxl",
        "split_subpath": "data_split/ptbxl/sub_class/ptbxl_sub_class_test.csv",
        "dataset_name": "ptbxl",
        # WPW: delta-wave 形态非常局部；LVH: 高 R 波，整段心搏都有
        "default_combos": [["WPW"], ["LVH"]],
    },
    "ptbxl-form": {
        "data_subdir": "ptbxl",
        "split_subpath": "data_split/ptbxl/form/ptbxl_form_test.csv",
        "dataset_name": "ptbxl",
        # PVC: 孤立宽 QRS 心搏（最戏剧性的时间局部事件）
        # STE: ST 段抬高（每拍 ST 区段都该点亮）
        "default_combos": [["PVC"], ["STE"]],
        # 每个 combo 用最具诊断意义的导联展示：
        #   PVC: Lead II (idx=1) — 节律导联，宽 QRS 心搏最直观
        #   STE: V2  (idx=7)    — 前壁/前侧壁 STE 抬高最显眼
        "default_leads": [1, 7],
    },
    "ptbxl-rhythm": {
        "data_subdir": "ptbxl",
        "split_subpath": "data_split/ptbxl/rhythm/ptbxl_rhythm_test.csv",
        "dataset_name": "ptbxl",
        # AFIB: 全程不规则；BIGU: 二联律（正常–PVC 交替）—— 一个全局一个交替
        "default_combos": [["AFIB"], ["BIGU"]],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="局部对齐 Attention 矩阵可视化")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints/best/vit_small_bestZeroShotAll_ckpt.pth"),
        help="完整 TALE checkpoint 路径",
    )
    parser.add_argument(
        "--ckpt-ablation",
        type=str,
        default=None,
        help=(
            "消融 checkpoint（w/o local loss）路径。提供时图变为 3 行 x 2 列布局："
            "ECG / 完整 TALE / 消融。"
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT.parent / "datasets" / "finetune"),
        help="下游数据集根目录",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ptbxl-form",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="可视化使用的 PTB-XL 子任务，默认 ptbxl-form (PVC+STE 局部最戏剧)",
    )
    parser.add_argument(
        "--combinations",
        type=str,
        default=None,
        help=(
            "覆盖默认标签组合，JSON 形式，如 '[[\"PVC\"],[\"STE\"]]'。"
            "若不传，使用所选 dataset 的默认 dramatic preset。"
        ),
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default=None,
        help="测试集 CSV 路径（默认按 --dataset 自动拼接）",
    )
    parser.add_argument(
        "--prompt_json",
        type=str,
        default=str(PROJECT_ROOT / "zeroshot" / "CKEPE_prompt.json"),
        help="CKEPE prompt 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "第三篇论文" / "figures" / "fig-attention-map.png"),
        help="输出图片路径",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(ckpt_path: str, device: str) -> ECGCLIP:
    """加载 ECGCLIP 模型（ViT-Small backbone）"""
    network_config = {
        "ecg_model": "vit_small",
        "num_leads": 12,
        "text_model": "ncbi/MedCPT-Query-Encoder",
        "free_layers": 12,
        "feature_dim": 768,
        "projection_head": {
            "mlp_hidden_size": 256,
            "projection_size": 256,
        },
        "use_jepa_init": False,  # 加载完整 ckpt，不需要单独加载 JEPA
    }
    model = ECGCLIP(network_config)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 去除 DDP 前缀 module.
    state_dict = {
        k.replace("module.", ""): v for k, v in ckpt.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model


def select_multilabel_samples(
    dataset: ECGDataset,
    combinations: list[list[str]],
    seed: int = 42,
    prefer_clean: bool = True,
) -> dict[str, dict]:
    """
    为每个标签组合选取一个同时包含所有指定标签的样例。

    Args:
        prefer_clean: 优先选"label 总数最少"的样例（最接近"只含目标 combo"的纯样本），
                      让 attention 不被无关合并疾病干扰。
    """
    rng = np.random.RandomState(seed)
    labels = dataset.labels
    label_names = dataset.labels_name
    total_count = labels.sum(axis=1)

    selected = {}
    for combo in combinations:
        mask = np.ones(len(labels), dtype=bool)
        for cls_name in combo:
            cls_idx = label_names.index(cls_name)
            mask &= (labels[:, cls_idx] == 1)
        candidates = np.where(mask)[0]

        if len(candidates) == 0:
            raise ValueError(f"找不到同时包含 {combo} 的样例，请调整 TARGET_COMBINATIONS")

        if prefer_clean and len(candidates) > 1:
            cand_counts = total_count[candidates]
            min_count = int(cand_counts.min())
            cleanest = candidates[cand_counts == min_count]
            print(
                f"  组合 {combo}: {len(candidates)} 个候选, "
                f"最干净的 {len(cleanest)} 个 (合计标签数={min_count})"
            )
            candidates = cleanest

        idx = rng.choice(candidates)
        combo_key = "+".join(combo)
        selected[combo_key] = {"idx": int(idx), "labels": combo}
        actual = [label_names[j] for j in range(labels.shape[1]) if labels[idx, j] == 1]
        print(f"  组合 '{combo_key}': 样例索引 {idx}，实际标签 {actual}")

    return selected


def extract_attention(
    model: ECGCLIP,
    ecg: torch.Tensor,
    text: str,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """
    提取单个 ECG--文本对的 attention 矩阵。

    返回:
        attn_map: [num_words, num_patches] 的 attention 权重
        word_labels: 每个 word token 对应的文本
    """
    model.eval()
    with torch.no_grad():
        ecg = ecg.unsqueeze(0).to(device)  # [1, 12, 5000]

        # ---- ECG token 嵌入 ----
        ecg_tokens = model.get_local_ecg_token_emb(ecg)  # [1, N_patches, 256]

        # ---- 文本 token 嵌入 ----
        tokenized = model._tokenize([text])
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        text_tokens, text_mask = model.get_text_token_emb(
            input_ids, attention_mask, return_mask=True
        )  # [1, seq_len, 256], [1, seq_len]

        # ---- 解码 token 文本标签 ----
        token_ids = input_ids[0].cpu().tolist()
        decoded_tokens = model.tokenizer.convert_ids_to_tokens(token_ids)
        mask_np = text_mask[0].cpu().numpy().astype(bool)

        # 过滤有效 token
        valid_tokens = [t for t, m in zip(decoded_tokens, mask_np) if m]

        # ---- 计算 attention（复用 gloria_attention_fn_1d 的逻辑）----
        # context: [1, dim, N_patches]
        context = ecg_tokens.permute(0, 2, 1).contiguous()
        # word: [1, dim, N_words]
        word_emb = text_tokens[:, mask_np, :].permute(0, 2, 1).contiguous()

        n_words = word_emb.shape[2]
        n_patches = context.shape[2]

        # attention: context^T @ word -> [N_patches, N_words]
        contextT = context.squeeze(0).T  # [N_patches, dim]
        wordM = word_emb.squeeze(0)       # [dim, N_words]
        raw_attn = contextT @ wordM       # [N_patches, N_words]

        # softmax over patches (for each word, which patches are important)
        raw_attn = raw_attn * TEMP1
        attn_map = torch.softmax(raw_attn, dim=0)  # [N_patches, N_words]
        attn_map = attn_map.T  # [N_words, N_patches]

    return attn_map.cpu().numpy(), valid_tokens


def fix_hyphenated_words(
    attn_map: np.ndarray, tokens: list[str]
) -> tuple[np.ndarray, list[str]]:
    """
    将 word - word 形式的连字符组合还原为单词（如 long-standing）。
    连字符前后的 token 及中间的 "-" 合并为一个 token，attention 取平均。
    """
    result_attn = []
    result_tokens = []
    i = 0
    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and tokens[i + 1] == "-"
            and not tokens[i].startswith("##")
            and not tokens[i + 2].startswith("##")
        ):
            merged_token = tokens[i] + "-" + tokens[i + 2]
            merged_attn = np.mean(attn_map[[i, i + 1, i + 2]], axis=0)
            result_tokens.append(merged_token)
            result_attn.append(merged_attn)
            i += 3
        else:
            result_tokens.append(tokens[i])
            result_attn.append(attn_map[i])
            i += 1
    return np.stack(result_attn, axis=0), result_tokens


def merge_subwords(attn_map: np.ndarray, tokens: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    合并 BPE subword token 的 attention 权重（取平均）。

    返回:
        merged_attn: [num_merged_words, num_patches]
        merged_labels: 合并后的词语列表
    """
    merged_attn = []
    merged_labels = []
    current_word_parts = []
    current_attns = []

    for i, token in enumerate(tokens):
        if token.startswith("##"):
            # subword，合并到前一个词
            current_word_parts.append(token[2:])
            current_attns.append(attn_map[i])
        else:
            # 新词开始：先保存前一个词
            if current_word_parts:
                merged_labels.append("".join(current_word_parts))
                merged_attn.append(np.mean(current_attns, axis=0))
            current_word_parts = [token]
            current_attns = [attn_map[i]]

    # 保存最后一个词
    if current_word_parts:
        merged_labels.append("".join(current_word_parts))
        merged_attn.append(np.mean(current_attns, axis=0))

    return np.stack(merged_attn, axis=0), merged_labels


def deduplicate_words(
    attn_map: np.ndarray,
    word_labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    合并重复词语的 attention 权重（取平均）。
    同一个词在 CKEPE prompt 中多次出现时，将其 attention 合并为一条。
    """
    from collections import OrderedDict

    word_groups: OrderedDict[str, list[int]] = OrderedDict()
    for i, word in enumerate(word_labels):
        key = word.lower().strip()
        if key not in word_groups:
            word_groups[key] = []
        word_groups[key].append(i)

    merged_attn = []
    merged_labels = []
    for key, indices in word_groups.items():
        merged_attn.append(np.mean(attn_map[indices], axis=0))
        # 使用首次出现的原始大小写
        merged_labels.append(word_labels[indices[0]])

    return np.stack(merged_attn, axis=0), merged_labels


_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "with", "by", "from", "as", "it", "its",
    "that", "this", "be", "has", "have", "had", "not", "no", "but", "if",
    ",", ".", ";", ":", "-", "(", ")", "/", "type", "related", "i", "ii",
    "non", "specific", "specifc", "abnormal", "normal", "associated",
    "including", "without", "with", "due", "other", "also", "patient",
    "rhythm", "node",
    "standing",
}


def select_top_words(
    attn_map: np.ndarray,
    word_labels: list[str],
    max_words: int = 12,
) -> tuple[list[int], list[str]]:
    """去重 → 去停用词 → 按 attention 峰值取 top-K，返回索引到原数组的位置。

    返回值:
        keep_indices_in_dedup: 在 deduplicated 序列中要保留的索引（已排序）
        merged_labels:         deduplicated 的完整 word_labels（用于 indexing）
    """
    attn_dedup, labels_dedup = deduplicate_words(attn_map, word_labels)

    keep = [
        i for i, w in enumerate(labels_dedup)
        if w.lower().strip() not in _STOPWORDS and len(w.strip()) > 1
    ]
    if not keep:
        keep = list(range(len(labels_dedup)))

    if len(keep) > max_words:
        peak = attn_dedup[keep].max(axis=1)
        order = np.argsort(peak)[-max_words:]
        keep = sorted(int(keep[i]) for i in order)

    return keep, labels_dedup


def apply_word_indices(
    attn_map: np.ndarray,
    word_labels: list[str],
    keep_indices: list[int],
    dedup_labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    """用 ``keep_indices``（基于 deduplicated 序列）从 ``attn_map`` 抽出对应行。

    要求 ``attn_map`` / ``word_labels`` 与 ``dedup_labels`` 来自同一 tokenization
    （token 数和顺序一致）；deduplicate_words 是确定性的，所以两次调用的输出索引可对齐。
    """
    attn_dedup, _ = deduplicate_words(attn_map, word_labels)
    return attn_dedup[keep_indices], [dedup_labels[i] for i in keep_indices]


def filter_diagnostic_words(
    attn_map: np.ndarray,
    word_labels: list[str],
    max_words: int = 12,
) -> tuple[np.ndarray, list[str]]:
    """单模型版本：等价于 select_top_words + apply_word_indices."""
    keep, dedup_labels = select_top_words(attn_map, word_labels, max_words)
    return apply_word_indices(attn_map, word_labels, keep, dedup_labels)


def aggregate_patches(attn_map: np.ndarray, seg_size: int) -> np.ndarray:
    """
    将 [num_words, num_patches] 的 attention 按 seg_size 个 patch 聚合为 segment。
    每个 segment 的 attention 为组内 patch 的均值。
    """
    n_words, n_patches = attn_map.shape
    n_segs = n_patches // seg_size
    # 截断到整数倍
    trimmed = attn_map[:, :n_segs * seg_size]
    # reshape 并求均值
    return trimmed.reshape(n_words, n_segs, seg_size).mean(axis=2)


_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_attention_maps_compare(
    attn_maps_full: list[np.ndarray],
    attn_maps_abl: list[np.ndarray],
    word_labels_list: list[list[str]],
    ecg_signals: list[np.ndarray],
    class_names: list[str],
    n_patches: int,
    output_path: str,
    row_titles: tuple[str, str] = ("TALE (full)", "w/o local loss"),
    ecg_leads: list[int] | None = None,   # 每列各自的导联索引
    seg_size: int = SEGMENT_SIZE,
):
    """3 行 x 2 列：ECG / 完整 TALE / 消融。两侧共用同一组诊断词。

    Style: IEEE Transactions 友好 —— 色盲安全的 viridis colormap、
    300 dpi、Arial sans-serif、字号在 7-11 pt 之间（双栏可读）。
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # IEEE 要求 TrueType（避免 Type-3 字体被退稿）
        "ps.fonttype": 42,
    })

    # OrRd: ColorBrewer 9-class Orange-Red，Nature/Cell 常用红系
    # 从奶黄 → 橙 → 深红，层次比 Reds 丰富；单调递增 → 灰度打印仍有效
    CMAP = "OrRd"
    ECG_COLOR = "#1a1a1a"     # 接近黑但更柔和

    fig = plt.figure(figsize=(7.16, 5.2))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 2.5, 2.5],
        hspace=0.18,
        wspace=0.55,
    )

    time_signal = np.linspace(0, 10, 5000)
    n_segs = n_patches // seg_size
    if ecg_leads is None:
        ecg_leads = [1, 1]   # 默认全部 Lead II

    for sample_idx in range(2):
        words = word_labels_list[sample_idx]
        ecg_raw = ecg_signals[sample_idx]
        cls_name = class_names[sample_idx]
        col = sample_idx
        lead_idx = ecg_leads[sample_idx]
        lead_name = _LEAD_NAMES[lead_idx] if 0 <= lead_idx < len(_LEAD_NAMES) else f"Ch.{lead_idx}"

        # ---- 找完整模型 attention 在哪个时间段最强（用于跨行垂直对齐线）----
        attn_full_seg = aggregate_patches(attn_maps_full[sample_idx], seg_size)
        attn_abl_seg = aggregate_patches(attn_maps_abl[sample_idx], seg_size)
        sample_vmax = float(max(attn_full_seg.max(), attn_abl_seg.max()))

        # 跨词平均后取相对峰值 (>= 70% of max) 的 segment 作为标记点
        mean_attn = attn_full_seg.mean(axis=0)
        peak_thresh = mean_attn.max() * 0.75
        peak_segs = np.where(mean_attn >= peak_thresh)[0]
        peak_times = [(s + 0.5) * (10.0 / n_segs) for s in peak_segs]

        # ---- Row 0: ECG ----
        ax_ecg = fig.add_subplot(gs[0, col])
        lead_signal = ecg_raw[lead_idx]
        ax_ecg.plot(time_signal, lead_signal, color=ECG_COLOR, linewidth=0.7)
        ax_ecg.set_xlim(0, 10)
        ax_ecg.tick_params(axis="x", labelbottom=False, length=2)
        ax_ecg.set_yticks([])
        # 每列单独显示导联名（PVC 看 II，STE 看 V2 等）
        ax_ecg.set_ylabel(f"Lead {lead_name}", fontsize=8, rotation=90, labelpad=4)
        for spine in ("top", "right", "left"):
            ax_ecg.spines[spine].set_visible(False)
        for s in range(1, n_segs):
            t = s * seg_size * (10.0 / n_patches)
            ax_ecg.axvline(t, color="#bdc3c7", linewidth=0.3, alpha=0.4)
        # 跨行对齐线：在 peak segment 上画一条贯穿三行的虚线
        for pt in peak_times:
            ax_ecg.axvline(pt, color="#1a5fb4", linestyle="--",
                           linewidth=0.8, alpha=0.7, zorder=3)
        div_ecg = make_axes_locatable(ax_ecg)
        dummy = div_ecg.append_axes("right", size="5%", pad=0.05)
        dummy.set_visible(False)

        for row_idx, (attn_seg, row_title) in enumerate(zip(
            (attn_full_seg, attn_abl_seg), row_titles,
        )):
            ax_attn = fig.add_subplot(gs[1 + row_idx, col], sharex=ax_ecg)
            im = ax_attn.imshow(
                attn_seg,
                aspect="auto",
                cmap=CMAP,
                interpolation="nearest",
                vmin=0.0, vmax=sample_vmax,
                extent=[0, 10, len(words) - 0.5, -0.5],
            )
            xticks = np.linspace(0, 10, 6)
            ax_attn.set_xticks(xticks)
            ax_attn.set_xticklabels([f"{t:.0f}" for t in xticks], fontsize=7)
            ax_attn.set_xlim(0, 10)
            ax_attn.set_yticks(range(len(words)))
            ax_attn.set_yticklabels(words, fontsize=7)
            # 只在最左列（col == 0）显示行标题；右列共享同一行含义，省掉避免重复
            if col == 0:
                ax_attn.set_ylabel(row_title, fontsize=10, fontweight="bold", labelpad=8)
            if row_idx == 1:
                ax_attn.set_xlabel("Time (s)", fontsize=8, labelpad=2)
            else:
                ax_attn.tick_params(axis="x", labelbottom=False)

            # 跨行对齐线：在 peak segment 上画一条贯穿三行的虚线
            for pt in peak_times:
                ax_attn.axvline(pt, color="#1a5fb4", linestyle="--",
                                linewidth=0.8, alpha=0.7, zorder=3)

            div = make_axes_locatable(ax_attn)
            cax = div.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=5)
            # 三个挑选的刻度（0、中位、最大），直接用三位小数显示，避免 1e-3 前缀
            cb.set_ticks([0, sample_vmax / 2, sample_vmax])
            cb.set_ticklabels([f"{t:.3f}" for t in [0, sample_vmax / 2, sample_vmax]])

        # 子图组标题放在第二个热力图下方
        ax_attn.text(
            0.5, -0.32,
            f"({chr(97 + sample_idx)}) {cls_name}",
            transform=ax_attn.transAxes,
            fontweight="bold", fontsize=11,
            ha="center", va="top",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"对比图已保存至: {output_path}")
    plt.close(fig)


def plot_attention_maps(
    attn_maps: list[np.ndarray],
    word_labels_list: list[list[str]],
    ecg_signals: list[np.ndarray],
    class_names: list[str],
    n_patches: int,
    output_path: str,
    ecg_lead: int = 1,  # Lead II（索引 1），最常用的临床导联
    seg_size: int = SEGMENT_SIZE,
):
    """
    左右排列 2 个样例（双栏图），每列：ECG 波形（上）+ Attention 热力图（下）。
    子图标题放在各自热力图下方。
    布局：2 行 × 2 列，figsize=(7.16, 3.2)，适合 IEEE 双栏。
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
    })

    # 双栏宽度，左右两个样例
    fig = plt.figure(figsize=(7.16, 3.2))

    # 2 行 × 2 列：hspace 控制列内 ECG-Attn 间距，wspace 控制两列间距
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 2.5],
        hspace=0.08,   # 列内 ECG 与 Attn 紧凑
        wspace=0.55,   # 两列之间留足空间，防止 colorbar/ylabel 叠字
    )

    # ECG 采样率 500 Hz，信号长度 5000 点 = 10 秒
    time_signal = np.linspace(0, 10, 5000)
    n_segs = n_patches // seg_size

    for sample_idx in range(2):
        attn_raw = attn_maps[sample_idx]
        words = word_labels_list[sample_idx]
        ecg_raw = ecg_signals[sample_idx]  # [12, 5000]
        cls_name = class_names[sample_idx]

        attn = aggregate_patches(attn_raw, seg_size)
        print(f"  {cls_name}: {attn_raw.shape[1]} patches → {attn.shape[1]} segments")

        # per-row 归一化：让每个词在自身尺度内可视化，突出时间段分布差异
        row_min = attn.min(axis=1, keepdims=True)
        row_max = attn.max(axis=1, keepdims=True)
        denom = np.where(row_max - row_min > 1e-9, row_max - row_min, 1e-9)
        attn = (attn - row_min) / denom

        col = sample_idx  # 0（左）或 1（右）

        # ---- ECG 波形 ----
        ax_ecg = fig.add_subplot(gs[0, col])
        lead_signal = ecg_raw[ecg_lead]
        ax_ecg.plot(time_signal, lead_signal, color="#2c3e50", linewidth=0.6)
        ax_ecg.set_xlim(0, 10)
        ax_ecg.tick_params(axis="x", labelbottom=False, length=2)
        ax_ecg.set_yticks([])
        ax_ecg.text(
            -0.01, 0.5, "Lead II",
            transform=ax_ecg.transAxes,
            fontsize=7, rotation=90, va="center", ha="right",
        )
        ax_ecg.spines["top"].set_visible(False)
        ax_ecg.spines["right"].set_visible(False)
        ax_ecg.spines["left"].set_visible(False)

        # segment 边界竖线（ECG）
        for s in range(1, n_segs):
            t = s * seg_size * (10.0 / n_patches)
            ax_ecg.axvline(t, color="#bdc3c7", linewidth=0.3, alpha=0.4)

        # ECG 右侧占位轴（与 Attn colorbar 等宽，保证左边界对齐）
        div_ecg = make_axes_locatable(ax_ecg)
        dummy_ax = div_ecg.append_axes("right", size="5%", pad=0.05)
        dummy_ax.set_visible(False)

        # ---- Attention 矩阵 ----
        ax_attn = fig.add_subplot(gs[1, col], sharex=ax_ecg)
        im = ax_attn.imshow(
            attn,
            aspect="auto",
            cmap="Reds",
            interpolation="nearest",
            vmin=0.0, vmax=1.0,
            extent=[0, 10, len(words) - 0.5, -0.5],
        )

        # 横轴：时间
        xtick_values = np.linspace(0, 10, 6)
        ax_attn.set_xticks(xtick_values)
        ax_attn.set_xticklabels([f"{t:.0f}" for t in xtick_values], fontsize=7)
        ax_attn.set_xlabel("Time (s)", fontsize=8, labelpad=2)
        ax_attn.set_xlim(0, 10)

        # 纵轴：词语
        ax_attn.set_yticks(range(len(words)))
        ax_attn.set_yticklabels(words, fontsize=7)

        # colorbar
        div_attn = make_axes_locatable(ax_attn)
        cbar_ax = div_attn.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=5)

        # 标题放在热力图下方（x 轴外侧）
        ax_attn.text(
            0.5, -0.28,
            f"({chr(97 + sample_idx)}) {cls_name}",
            transform=ax_attn.transAxes,
            fontweight="bold", fontsize=11,
            ha="center", va="top",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"图片已保存至: {output_path}")
    plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("局部对齐 Attention 矩阵可视化")
    print("=" * 60)

    # ---- 1. 加载模型 ----
    print(f"\n[1/5] 加载完整模型: {args.ckpt}")
    model = load_model(args.ckpt, device)
    model_abl = None
    if args.ckpt_ablation:
        print(f"          消融模型: {args.ckpt_ablation}")
        model_abl = load_model(args.ckpt_ablation, device)

    # ---- 2. 加载数据集 ----
    cfg = DATASET_CONFIGS[args.dataset]
    print(f"\n[2/5] 加载数据集: {args.dataset}")
    data_path = os.path.join(args.data_root, cfg["data_subdir"])
    split_csv = args.split_csv or os.path.join(args.data_root, cfg["split_subpath"])
    dataset = ECGDataset(
        data_path=data_path,
        csv_file=split_csv,
        mode="test",
        dataset_name=cfg["dataset_name"],
    )
    print(f"  split_csv: {split_csv}")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  类别: {dataset.labels_name}")

    # ---- 3. 选取多标签样例 ----
    if args.combinations:
        import json as _json
        combos = _json.loads(args.combinations)
    else:
        combos = cfg["default_combos"]
    print(f"\n[3/5] 自动选取多标签样例: {combos}")
    selected = select_multilabel_samples(dataset, combos, seed=args.seed)

    # ---- 4. 加载 CKEPE prompt ----
    print(f"\n[4/5] 加载 CKEPE prompt")
    with open(args.prompt_json, "r") as f:
        prompt_dict = yaml.load(f, Loader=yaml.FullLoader)

    # ---- 5. 提取 attention 并绘图 ----
    print(f"\n[5/5] 提取 attention 矩阵并绘图")
    attn_maps = []
    attn_maps_abl = []
    word_labels_list = []
    ecg_signals = []
    panel_names = []

    for combo_key, info in selected.items():
        idx = info["idx"]
        combo_labels = info["labels"]
        ecg, target = dataset[idx]

        prompt_text = " ".join(prompt_dict[cls] for cls in combo_labels)
        print(f"\n  组合: {combo_key} (样例 #{idx})")
        print(f"  标签向量: {target.numpy()}")
        print(f"  拼接 Prompt: {prompt_text[:100]}...")

        # --- 完整 TALE: 提 attention 并锁定 top-K 诊断词 ---
        attn_full, tokens = extract_attention(model, ecg, prompt_text, device)
        attn_full, tokens = merge_subwords(attn_full, tokens)
        attn_full, tokens = fix_hyphenated_words(attn_full, tokens)
        # 6 个最 distinct 的诊断词足以体现局部对齐；之前 10 个里同义词重复严重
        keep, dedup_labels = select_top_words(attn_full, tokens, max_words=6)
        attn_full_keep, words_keep = apply_word_indices(
            attn_full, tokens, keep, dedup_labels
        )
        print(f"  完整模型词表: {words_keep}")

        attn_maps.append(attn_full_keep)
        word_labels_list.append(words_keep)
        ecg_signals.append(ecg.numpy())
        panel_names.append(combo_key)

        # --- 消融模型：复用同一组词索引，便于直接对比 ---
        if model_abl is not None:
            attn_abl, tokens_abl = extract_attention(model_abl, ecg, prompt_text, device)
            attn_abl, tokens_abl = merge_subwords(attn_abl, tokens_abl)
            attn_abl, tokens_abl = fix_hyphenated_words(attn_abl, tokens_abl)
            attn_abl_keep, _ = apply_word_indices(
                attn_abl, tokens_abl, keep, dedup_labels
            )
            attn_maps_abl.append(attn_abl_keep)

    n_patches = attn_maps[0].shape[1]
    # 每个 combo 对应的导联：优先 dataset config 的 default_leads，否则全 II
    ecg_leads = cfg.get("default_leads", [1] * len(combos))[: len(combos)]
    if model_abl is not None:
        plot_attention_maps_compare(
            attn_maps, attn_maps_abl, word_labels_list, ecg_signals,
            panel_names, n_patches, args.output, ecg_leads=ecg_leads,
        )
    else:
        plot_attention_maps(
            attn_maps, word_labels_list, ecg_signals, panel_names,
            n_patches, args.output,
        )

    print("\n完成!")


if __name__ == "__main__":
    main()
