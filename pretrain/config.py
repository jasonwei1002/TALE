from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_ROOT = PROJECT_ROOT.parent / 'datasets'  # 根据你的数据集位置调整

config = {
    "network": {
        "ecg_model": "vit_small",
        "num_leads": 12,
        # this part does not control builder/trainer
        "text_model": "ncbi/MedCPT-Query-Encoder",
        # "text_model": "neuml/pubmedbert-base-embeddings",
        "free_layers": 12,  # set 12 to freeze all layer in bert
        "feature_dim": 768,
        "projection_head": {
            "mlp_hidden_size": 256,
            "projection_size": 256,
        },
        # --- Ablation switches ---
        # Set False for "w/o JEPA init" variant
        "use_jepa_init": True,
    },
    "dataset": {
        "dataset_name": "mimic",
        "data_path": DATASETS_ROOT / "pretrain",
    },
    "trainer": {
        "batch_size": 256,
        "val_batch_size": 512,
        "max_epochs": 50,
        "num_workers": 12,
        "local_loss_weight": 0.5,
        "local_temp1": 4.0,
        "local_temp2": 5.0,
        "local_temp3": 10.0,
        "local_max_token_len": 256,
        "jaccard_t": 0.1,
        "soft_neg_scale": 0.3,
        "use_uma_loss": True,        # UMA from MERL (liu2024zero); keep True, not ablated
        # --- Ablation switches (our proposed components) ---
        "use_jaccard_mask": True,    # Set False for "w/o Jaccard soft-neg" variant
        "use_local_loss": True,      # Set False for "w/o local loss" variant
    },
    "optimizer": {
        "params": {
            "lr": 3e-4,
            "lr_text": 3e-5,
            "weight_decay": 1e-3,
        },
    },
    "zeroshot": {
        "prompt_type": "CKEPE",
        "prompt_dict": PROJECT_ROOT / "zeroshot/CKEPE_prompt.json",
        "meta_data_path": DATASETS_ROOT / "finetune",
        "meta_split_path": DATASETS_ROOT / "finetune/data_split",
        "batch_size": 256,
        "num_workers": 8,

        "val_sets": {
            "ptbxl_super_class": {
                "data_path": "ptbxl",
                "split_path": "ptbxl/super_class/ptbxl_super_class_val.csv",
            },
            "ptbxl_sub_class": {
                "data_path": "ptbxl",
                "split_path": "ptbxl/sub_class/ptbxl_sub_class_val.csv",
            },
            "ptbxl_form": {
                "data_path": "ptbxl",
                "split_path": "ptbxl/form/ptbxl_form_val.csv",
            },
            "ptbxl_rhythm": {
                "data_path": "ptbxl",
                "split_path": "ptbxl/rhythm/ptbxl_rhythm_val.csv",
            },
            "icbeb": {
                "data_path": "icbeb",
                "split_path": "icbeb/icbeb_val.csv",
            },
            "chapman": {
                "data_path": "",
                "split_path": "chapman/chapman_val.csv",
            },
        },
    },
    "swanlab_name": "vit_small",
}
