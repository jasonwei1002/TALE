import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq_signals_backbone.models.wav2vec2.wav2vec2_cmsc import (
    Wav2Vec2CMSCModel,
    Wav2Vec2CMSCConfig,
)


class ECGFMModel(nn.Module):
    def __init__(
        self,
        model_size: str = "small",
        shared_emb_dim: int = 256,
        num_leads: int = 12,
        proj: str = "linear",
        drop: float = 0.0,
        proj_bias: bool = False,
    ):
        """ECGFM encoder-only wrapper (Wav2Vec2CMSC backbone + pooling head)."""
        super().__init__()

        self.num_leads = num_leads

        if model_size == "small":
            self.encoder_embed_dim = 768
            self.encoder_attention_heads = 12
            self.encoder_layers = 8
            self.encoder_ffn_embed_dim = 3072
        elif model_size == "base":
            self.encoder_embed_dim = 768
            self.encoder_attention_heads = 12
            self.encoder_layers = 12
            self.encoder_ffn_embed_dim = 3072
        elif model_size == "large":
            self.encoder_embed_dim = 1024
            self.encoder_attention_heads = 16
            self.encoder_layers = 24
            self.encoder_ffn_embed_dim = 4096
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        self.init_ecg_encoder()

        prev_chs = self.ecg_encoder.cfg.encoder_embed_dim
        if proj == "linear":
            self.head = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(prev_chs, shared_emb_dim, bias=proj_bias),
            )
        elif proj == "mlp":
            self.head = nn.Sequential(
                nn.Linear(prev_chs, 2 * shared_emb_dim, bias=True),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(2 * shared_emb_dim, shared_emb_dim, bias=proj_bias),
            )
        else:
            raise ValueError(f"Unknown projection type: {proj}")

    def init_ecg_encoder(self):
        cfg = Wav2Vec2CMSCConfig(
            apply_mask=True,
            mask_prob=0.65,
            quantize_targets=True,
            final_dim=256,
            dropout_input=0.1,
            dropout_features=0.1,
            feature_grad_mult=0.1,
            encoder_embed_dim=self.encoder_embed_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            in_d=self.num_leads,
            encoder_layers=self.encoder_layers,
            encoder_ffn_embed_dim=self.encoder_ffn_embed_dim,
        )
        self.ecg_encoder = Wav2Vec2CMSCModel(cfg)

    def forward(self, ecg, return_tokens=False):
        if ecg.dim() == 4:
            b, c, n, d = ecg.shape
            ecg = ecg.reshape(b * c, n, d)
        elif ecg.dim() != 3:
            raise ValueError("Input tensor must be 3D or 4D")

        ecg_out = self.ecg_encoder(source=ecg.float(), mask=False, features_only=True)
        features = ecg_out["x"].float()

        if return_tokens:
            return features
        raw_features = features.mean(dim=1)

        return raw_features

if __name__ == "__main__":
    ecg = torch.randn(1, 12, 5000)
    model = ECGFMModel(model_size="small")
    output = model(ecg)
    print(output.shape)
