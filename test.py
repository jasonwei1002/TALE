import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch import nn
from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig

class LLM2VecWrapper(LLM2Vec):
    def prepare_for_tokenization(self, text):
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + text.strip()
            + "<|eot_id|>"
        )
        return  text
    
class LlamaVec_1B_FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        model_path = 'microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned'
        config = AutoConfig.from_pretrained(model_path)

        model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.l2v = LLM2VecWrapper(model, tokenizer, pooling_mode="mean", max_length=512, skip_instruction=True)
        
    def extract_features(self, text):
        with torch.amp.autocast('cuda'):
            reps_norm = self.l2v.encode(text)
            reps_norm = torch.nn.functional.normalize(reps_norm, p=2, dim=1)
        return {"embeds": reps_norm}

text_model = LlamaVec_1B_FeatureExtractor()
captions = ["this is a test"]
embeddings = text_model.extract_features(captions)
print(embeddings.shape)
