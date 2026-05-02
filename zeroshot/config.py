from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_ROOT = PROJECT_ROOT.parent / 'datasets'  # 根据你的数据集位置调整

config = {
    "network": {
        # "ecg_model": "resnet18",
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
    },
    "dataset": {
        "dataset_name": "mimic",
        "data_path": DATASETS_ROOT / "pretrain",
    },
    "trainer": {
        "batch_size": 1024,
        "val_batch_size": 512,
        "max_epochs": 100,
        "num_workers": 8,
    },
    "optimizer": {
        "params": {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-8,
        },
    },
    "zeroshot": {
        "prompt_type": "CKEPE",
        "prompt_dict": PROJECT_ROOT / "zeroshot/CKEPE_prompt.json",
        "meta_data_path": DATASETS_ROOT / 'finetune',
        "meta_split_path": DATASETS_ROOT / 'finetune/data_split',
        "batch_size": 256,
        "num_workers": 8,
        "test_sets": {
            "ptbxl_super_class": {
                "data_path": "ptbxl",
                "split_path": "ptbxl/super_class/ptbxl_super_class_test.csv",
            },
            "ptbxl_sub_class": {
                "data_path": "ptbxl",
                "split_path": "ptbxl/sub_class/ptbxl_sub_class_test.csv",
            },
            "ptbxl_form": {
                "data_path": "ptbxl",
                "split_path": "ptbxl/form/ptbxl_form_test.csv",
            },
            "ptbxl_rhythm": {
                "data_path": "ptbxl",
                "split_path": "ptbxl/rhythm/ptbxl_rhythm_test.csv",
            },
            "icbeb": {
                "data_path": "icbeb",
                "split_path": "icbeb/icbeb_test.csv",
            },
            "chapman": {
                "data_path": "",
                "split_path": "chapman/chapman_test.csv",
            },
        },
    },
    "wandb_name": "None",
}

