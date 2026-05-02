from cgi import test
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from vit1d import vit_small


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, spacial_dim + 1, embed_dim) / embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)        
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(0, 2, 1) # convert X shape (B, C, L) to (B, L, C)

        self.cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
        self.cls_tokens = self.cls_tokens.expand(x.shape[0], -1, -1) 
        x = torch.cat((self.cls_tokens, x), dim=1)
        x = x + self.positional_embedding[:, :, :].to(x.dtype)  # (L+1)NC
        x, att_map = self.mhsa(x[:, :1, :], x, x, average_attn_weights=True)
        x = self.c_proj(x)
        return x.squeeze(0), att_map[:, :, 1:]
    
class ECGCLIP(torch.nn.Module):
    def __init__(self, network_config):
        super(ECGCLIP, self).__init__()
        
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']

        # ecg signal encoder
        self.ecg_model = network_config['ecg_model']
        self.num_leads = network_config['num_leads']

        if 'resnet' in self.ecg_model:
            if self.ecg_model == 'resnet18':
                model = ResNet18()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_model == 'resnet34':
                model = ResNet34()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_model == 'resnet50':
                model = ResNet50()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_model == 'resnet101':
                model = ResNet101()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)

            self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)

        if ('vit' in self.ecg_model) or ('ecgfm' in self.ecg_model):
            if 'vit' in self.ecg_model:
                if self.ecg_model == 'vit_small':
                    model = vit_small()

                self.proj_e_input = model.width
            else:
                try:
                    from ecgfm import ECGFMModel
                except Exception as exc:
                    raise ImportError(
                        "Failed to import ECGFMModel. Please ensure ecgfm dependencies are installed."
                    ) from exc

                model_size = "small"
                if self.ecg_model in ("ecgfm_small", "ecgfm_base", "ecgfm_large"):
                    model_size = self.ecg_model.split("_", 1)[1]
                elif self.ecg_model != "ecgfm":
                    raise ValueError(f"Unknown ecgfm model variant: {self.ecg_model}")

                model = ECGFMModel(
                    model_size=model_size,
                    num_leads=self.num_leads,
                )
                self.proj_e_input = model.encoder_embed_dim

            self.proj_e = nn.Sequential(
                nn.Linear(self.proj_e_input, self.proj_hidden),
                nn.BatchNorm1d(self.proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden, self.proj_out),
                nn.BatchNorm1d(self.proj_out),
            )
            self.linear1 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)

        self.ecg_encoder = model
        use_jepa_init = network_config.get("use_jepa_init", True)
        if self.ecg_model == "vit_small" and use_jepa_init:
            ckpt = torch.load("../pretrain-ckpt/best_chkpt.pt", map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            target_state = {
                key[len("target_encoder."):]: value
                for key, value in state_dict.items()
                if key.startswith("target_encoder.")
            }
            self.ecg_encoder.load_state_dict(target_state, strict=False)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        # text encoder
        url = network_config['text_model']
        self.lm_model = AutoModel.from_pretrained(
            url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True, revision='main')


        # text projector
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )
        
        
    def _tokenize(self, text):
        # tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
        #                                                     add_special_tokens=True,
        #                                                     truncation=True,
        #                                                     max_length=256,
        #                                                     padding='max_length',
        #                                                     return_tensors='pt')
        tokenizer_output = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        return tokenizer_output

    # Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def meanpooling(output, mask):
        embeddings = output[0] # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    @torch.no_grad()
    def ext_ecg_emb(self, ecg):

        if 'resnet' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            ecg_emb = self.downconv(ecg_emb)
            proj_ecg_emb, att_map = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

        if 'vit' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)

        if 'ecgfm' in self.ecg_model:
            # ecg_in = self._prepare_ecgfm_input(ecg)
            ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)

        return proj_ecg_emb
    
    # @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        # text_emb = self.lm_model(input_ids=input_ids,
        #                          attention_mask=attention_mask).pooler_output
        output = self.lm_model(input_ids=input_ids,
                               attention_mask=attention_mask)
        
        text_emb = self.meanpooling(output, attention_mask)
        return text_emb

    def _build_text_token_emb(self, lm_output, input_ids, attention_mask, max_token_len=None):
        token_emb = lm_output[0]

        token_mask = attention_mask.to(dtype=torch.bool)
        special_ids = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        ]
        for sid in special_ids:
            if sid is not None:
                token_mask = token_mask & (input_ids != sid)

        if max_token_len is not None and token_emb.size(1) > max_token_len:
            token_emb = token_emb[:, :max_token_len, :]
            token_mask = token_mask[:, :max_token_len]

        bsz, seq_len, dim = token_emb.shape
        token_emb = token_emb.reshape(bsz * seq_len, dim)
        token_emb = self.proj_t(token_emb)
        token_emb = token_emb.reshape(bsz, seq_len, -1)
        token_emb = normalize(token_emb, dim=-1)
        return token_emb, token_mask

    def get_text_token_emb(self, input_ids, attention_mask, max_token_len=None, return_mask=False):
        output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask)
        token_emb, token_mask = self._build_text_token_emb(
            output, input_ids, attention_mask, max_token_len=max_token_len
        )
        if return_mask:
            return token_emb, token_mask
        return token_emb

    def get_ecg_tokens(self, ecg):
        if 'resnet' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            ecg_emb = self.downconv(ecg_emb)
            tokens = ecg_emb.transpose(1, 2)
            return tokens

        if 'vit' in self.ecg_model:
            tokens = self.ecg_encoder(ecg, return_tokens=True)
            return tokens

        if 'ecgfm' in self.ecg_model:
            tokens = self.ecg_encoder(ecg, return_tokens=True)
            return tokens

        raise ValueError(f"Unknown ecg model: {self.ecg_model}")

    def get_local_ecg_emb(self, ecg, num_segments):
        tokens = self.get_ecg_tokens(ecg)
        if ('vit' in self.ecg_model) or ('ecgfm' in self.ecg_model):
            bsz, seq_len, dim = tokens.shape
            tokens = tokens.reshape(bsz * seq_len, dim)
            tokens = self.proj_e(tokens)
            tokens = tokens.reshape(bsz, seq_len, -1)
        tokens = normalize(tokens, dim=-1)
        seg = F.adaptive_avg_pool1d(tokens.transpose(1, 2), num_segments).transpose(1, 2)
        seg = normalize(seg, dim=-1)
        return seg

    def get_local_ecg_token_emb(self, ecg):
        tokens = self.get_ecg_tokens(ecg)
        if ('vit' in self.ecg_model) or ('ecgfm' in self.ecg_model):
            bsz, seq_len, dim = tokens.shape
            tokens = tokens.reshape(bsz * seq_len, dim)
            tokens = self.proj_e(tokens)
            tokens = tokens.reshape(bsz, seq_len, -1)
        tokens = normalize(tokens, dim=-1)
        return tokens

    
    def forward(self, ecg, input_ids, attention_mask, return_text_tokens=False, max_token_len=None):
        ecg_token_emb = None
        if 'resnet' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)

        if 'resnet' in self.ecg_model:
            # attention pooling (only for resnet models)
            ecg_emb = self.downconv(ecg_emb)
            if return_text_tokens:
                ecg_token_emb = normalize(ecg_emb.transpose(1, 2), dim=-1)
            proj_ecg_emb, _ = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

            ecg_emb = self.avgpool(ecg_emb).view(ecg_emb.shape[0], -1)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))
        
        if 'vit' in self.ecg_model:
            if return_text_tokens:
                ecg_tokens = self.ecg_encoder(ecg, return_tokens=True)
                ecg_emb = ecg_tokens.mean(dim=1)
                bsz, seq_len, dim = ecg_tokens.shape
                ecg_token_emb = self.proj_e(ecg_tokens.reshape(bsz * seq_len, dim))
                ecg_token_emb = ecg_token_emb.reshape(bsz, seq_len, -1)
                ecg_token_emb = normalize(ecg_token_emb, dim=-1)
            else:
                ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))

        if 'ecgfm' in self.ecg_model:
            if return_text_tokens:
                ecg_tokens = self.ecg_encoder(ecg, return_tokens=True)
                ecg_emb = ecg_tokens.mean(dim=1)
                bsz, seq_len, dim = ecg_tokens.shape
                ecg_token_emb = self.proj_e(ecg_tokens.reshape(bsz * seq_len, dim))
                ecg_token_emb = ecg_token_emb.reshape(bsz, seq_len, -1)
                ecg_token_emb = normalize(ecg_token_emb, dim=-1)
            else:
                ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)

            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))

        proj_ecg_emb = normalize(proj_ecg_emb, dim=-1)


        # get text feature
        # text feature extraction is independent of the type of ecg encoder
        text_output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.meanpooling(text_output, attention_mask)
        proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = normalize(proj_text_emb, dim=-1)                                     

        text_token_emb = None
        text_token_mask = None
        if return_text_tokens:
            text_token_emb, text_token_mask = self._build_text_token_emb(
                text_output,
                input_ids,
                attention_mask,
                max_token_len=max_token_len,
            )

        if self.training:
            output_dict = {'ecg_emb': [ecg_emb1, ecg_emb2],
                           'proj_ecg_emb': [proj_ecg_emb],
                           'proj_text_emb': [proj_text_emb]}
            if return_text_tokens:
                output_dict.update({
                    'text_token_emb': text_token_emb,
                    'text_token_mask': text_token_mask,
                    'ecg_token_emb': ecg_token_emb,
                })
            return output_dict
        else:
            output_dict = {'ecg_emb': [ecg_emb1, ecg_emb2],
                           'proj_ecg_emb': [proj_ecg_emb],
                           'proj_text_emb': [proj_text_emb]}
            if return_text_tokens:
                output_dict.update({
                    'text_token_emb': text_token_emb,
                    'text_token_mask': text_token_mask,
                    'ecg_token_emb': ecg_token_emb,
                })
            return output_dict
