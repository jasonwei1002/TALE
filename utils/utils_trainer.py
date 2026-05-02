import os
import torch
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as torch_dist
import torch.distributed as dist
import pytz
from datetime import datetime
from tqdm import tqdm
import numpy as np
import yaml as yaml

from utils_loss import clip_loss, clip_loss_jaccard, gloria_local_loss
from zeroshot_val import zeroshot_eval
from scheduler import CosineAnnealingWarmupRestarts
import swanlab

class Trainer:
    def __init__(self, model,
                 optimizer, device, model_name, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.train_batch_size = args['batch_size']
        self.max_epochs = args['max_epochs']
        self.num_workers = args['num_workers']
        self.val_batch_size = args['val_batch_size']
        self.local_loss_weight = args.get('local_loss_weight', 1.0)
        self.local_temp1 = args.get('local_temp1', 4.0)
        self.local_temp2 = args.get('local_temp2', 5.0)
        self.local_temp3 = args.get('local_temp3', 10.0)
        self.local_max_token_len = args.get('local_max_token_len', 256)
        self.use_jaccard_mask = args.get('use_jaccard_mask', False)
        self.jaccard_t = args.get('jaccard_t', 0.2)
        self.soft_neg_scale = args.get('soft_neg_scale', 0.5)
        self.use_uma_loss = args.get('use_uma_loss', True)
        self.use_local_loss = args.get('use_local_loss', True)

    @staticmethod
    def _split_sentences(reports):
        sentence_lists = []
        for text in reports:
            if not isinstance(text, str):
                text = str(text)
            sentences = []
            for seg in text.split('.'):
                seg = seg.strip()
                if not seg:
                    continue
                sentences.append(seg)
            if len(sentences) == 0:
                fallback = text.strip()
                if fallback:
                    sentences = [fallback]
                else:
                    sentences = [" "]
            sentence_lists.append(sentences)
        max_sent = max(len(sents) for sents in sentence_lists)
        sent_mask = torch.zeros(len(sentence_lists), max_sent, dtype=torch.bool)
        flat_sentences = []
        for i, sents in enumerate(sentence_lists):
            sent_mask[i, :len(sents)] = True
            flat_sentences.extend(sents)
        return flat_sentences, sent_mask, sentence_lists

    def _sentence_emb_from_tokenize(self, reports):
        flat_sentences, sent_mask, sentence_lists = self._split_sentences(reports)
        sent_tokenize_output = self.model.module._tokenize(flat_sentences)
        sent_input_ids = sent_tokenize_output.input_ids.to(self.device).contiguous()
        sent_attention_mask = sent_tokenize_output.attention_mask.to(self.device).contiguous()
        sent_emb = self.model.module.get_text_emb(sent_input_ids, sent_attention_mask)
        sent_emb = self.model.module.proj_t(sent_emb)
        sent_emb = torch.nn.functional.normalize(sent_emb, dim=-1)
        max_sent = sent_mask.shape[1]
        sent_emb_3d = torch.zeros(
            len(sentence_lists),
            max_sent,
            sent_emb.shape[-1],
            device=sent_emb.device,
            dtype=sent_emb.dtype,
        )
        idx = 0
        for i, sents in enumerate(sentence_lists):
            cnt = len(sents)
            sent_emb_3d[i, :cnt] = sent_emb[idx:idx + cnt]
            idx += cnt
        return sent_emb_3d, sent_mask.to(device=self.device)


    # traing process
    def fit(self, train_dataset, val_dataset, args_zeroshot_eval):
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size,
                                  num_workers=self.num_workers, persistent_workers=True,
                                  drop_last=True, shuffle=False, pin_memory=True,
                                  sampler=DistributedSampler(train_dataset))
        
        val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size,
                                num_workers=self.num_workers, persistent_workers=True,
                                drop_last=True, shuffle=False, pin_memory=True,
                                sampler=DistributedSampler(val_dataset))
                                
        #UTC+8 time
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")
        # Allow env override for sweep runs (folder named by param config)
        model_checkpoints_folder = os.environ.get(
            "SWEEP_CKPT_FOLDER",
            os.path.join(f'../checkpoints/{current_time}')
        )
        if self.device == 0:
            if not os.path.exists(model_checkpoints_folder):
                print('create directory "{}" for save checkpoint!'.format(
                    model_checkpoints_folder))
                print('---------------------------')
                os.makedirs(model_checkpoints_folder)
            else:
                print('directory "{}" existing for save checkpoint!'.format(
                    model_checkpoints_folder))

        print('#########################################')
        print('Start training')
        print('#########################################')


        # scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=5000,
        #     T_mult=1,
        #     eta_min=1e-8,
        # )
        steps_per_epoch = len(train_loader)
        scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=steps_per_epoch*self.max_epochs,            
            max_lr=self.optimizer.param_groups[0]["lr"], 
            min_lr=1e-6,
            warmup_steps=int(0.1*self.max_epochs* steps_per_epoch),      
            gamma=1,                                   
        )

        skip_scheduler = False
        scaler = GradScaler()


        best_auc = 0
        patience = 5
        no_improve_count = 0
        is_main = (self.device == 0)
        for epoch_counter in tqdm(range(self.max_epochs), disable=not is_main):
            train_sampler = getattr(train_loader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch_counter)

            val_sampler = getattr(val_loader, "sampler", None)
            if hasattr(val_sampler, "set_epoch"):
                val_sampler.set_epoch(epoch_counter)

            epoch_loss = 0
            epoch_acc1 = []
            epoch_acc5 = []
            self.model.train()
            for data in tqdm(train_loader, disable=not is_main):
                self.model.train()
                # get raw text
                report = data['report']

                # get ecg
                ecg = data['ecg'].to(torch.float32).to(
                    self.device).contiguous()
                labels = data.get('labels', None)
                if self.use_jaccard_mask:
                    if labels is None:
                        raise KeyError('labels not found in dataset, but use_jaccard_mask is True.')
                    if not torch.is_tensor(labels):
                        labels = torch.tensor(labels)
                    labels = labels.to(torch.float32).to(self.device).contiguous()
                
                self.optimizer.zero_grad()

                with autocast():
                    report_tokenize_output = self.model.module._tokenize(report)

                    input_ids = report_tokenize_output.input_ids.to(
                        self.device).contiguous()
                    attention_mask = report_tokenize_output.attention_mask.to(
                        self.device).contiguous()

                    output_dict = self.model(
                        ecg,
                        input_ids,
                        attention_mask,
                        return_text_tokens=self.use_local_loss,
                        max_token_len=self.local_max_token_len,
                    )
                    ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'],\
                                                            output_dict['proj_ecg_emb'],\
                                                            output_dict['proj_text_emb']


                    world_size = torch_dist.get_world_size()
                    with torch.no_grad():
                        agg_proj_img_emb = [torch.zeros_like(proj_ecg_emb[0]) for _ in range(world_size)]
                        agg_proj_text_emb = [torch.zeros_like(proj_text_emb[0]) for _ in range(world_size)]

                        dist.all_gather(agg_proj_img_emb, proj_ecg_emb[0])
                        dist.all_gather(agg_proj_text_emb, proj_text_emb[0])

                        if self.use_uma_loss:
                            agg_proj_ecg_emb1 = [torch.zeros_like(ecg_emb[0]) for _ in range(world_size)]
                            agg_proj_ecg_emb2 = [torch.zeros_like(ecg_emb[1]) for _ in range(world_size)]
                            dist.all_gather(agg_proj_ecg_emb1, ecg_emb[0])
                            dist.all_gather(agg_proj_ecg_emb2, ecg_emb[1])
                        if self.use_jaccard_mask:
                            agg_labels = [torch.zeros_like(labels) for _ in range(world_size)]
                            dist.all_gather(agg_labels, labels)
                        # get current rank
                        rank = torch_dist.get_rank()

                    agg_proj_img_emb[rank] = proj_ecg_emb[0]
                    agg_proj_text_emb[rank] = proj_text_emb[0]

                    if self.use_uma_loss:
                        agg_proj_ecg_emb1[rank] = ecg_emb[0]
                        agg_proj_ecg_emb2[rank] = ecg_emb[1]
                    if self.use_jaccard_mask:
                        agg_labels[rank] = labels

                    agg_proj_img_emb = torch.cat(agg_proj_img_emb, dim=0)
                    agg_proj_text_emb = torch.cat(agg_proj_text_emb, dim=0)

                    if self.use_uma_loss:
                        agg_proj_ecg_emb1 = torch.cat(agg_proj_ecg_emb1, dim=0)
                        agg_proj_ecg_emb2 = torch.cat(agg_proj_ecg_emb2, dim=0)
                    if self.use_jaccard_mask:
                        agg_labels = torch.cat(agg_labels, dim=0)

                    if self.use_jaccard_mask:
                        cma_loss, acc1, acc5 = clip_loss_jaccard(
                            agg_proj_img_emb,
                            agg_proj_text_emb,
                            agg_labels,
                            jaccard_t=self.jaccard_t,
                            soft_neg_scale=self.soft_neg_scale,
                            device=self.device,
                        )
                        if self.use_uma_loss:
                            uma_loss, _, _ = clip_loss_jaccard(
                                agg_proj_ecg_emb1,
                                agg_proj_ecg_emb2,
                                agg_labels,
                                jaccard_t=self.jaccard_t,
                                soft_neg_scale=self.soft_neg_scale,
                                device=self.device,
                            )
                        else:
                            uma_loss = torch.zeros(1, device=self.device).squeeze()
                    else:
                        cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)
                        if self.use_uma_loss:
                            uma_loss, _, _ = clip_loss(agg_proj_ecg_emb1, agg_proj_ecg_emb2, device=self.device)
                        else:
                            uma_loss = torch.zeros(1, device=self.device).squeeze()

                    if self.use_local_loss:
                        text_token_emb = output_dict['text_token_emb']
                        text_token_mask = output_dict['text_token_mask']
                        ecg_token_emb = output_dict['ecg_token_emb']
                        local_loss = gloria_local_loss(
                            ecg_token_emb,
                            text_token_emb,
                            text_token_mask,
                            temp1=self.local_temp1,
                            temp2=self.local_temp2,
                            temp3=self.local_temp3,
                        )
                    else:
                        local_loss = torch.zeros(1, device=self.device).squeeze()

                    loss = cma_loss + uma_loss + self.local_loss_weight * local_loss

                    if self.device == 0:
                        swanlab.log({
                            'train/train_step_uma_loss': uma_loss.item(),
                            'train/train_step_cma_loss': cma_loss.item(),
                            'train/train_step_local_loss': local_loss.item(),
                            'train/train_step_total_loss': loss.item(),
                            'train/train_step_acc1': acc1.item(),
                            'train/train_step_acc5': acc5.item()}
                            )
                        
                    # accumalate loss for logging
                    epoch_loss += loss.item()
                    epoch_acc1.append(acc1.item())
                    epoch_acc5.append(acc5.item())
                    if self.device == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        swanlab.log({
                            'train/lr': lr
                            })
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if not skip_scheduler:
                        scheduler.step()

            # eval stage
            val_log = self.val(val_loader)
            
            if self.device == 0:
                # average train metric
                epoch_acc1 = np.array(epoch_acc1).mean()
                epoch_acc5 = np.array(epoch_acc5).mean()

                epoch_iter = len(train_loader)
                print(f'{epoch_counter} epoch loss is {epoch_loss/epoch_iter},\
                                    acc1 is {epoch_acc1}, acc5 is {epoch_acc5}')
                
                # log train and val epoch metric
                swanlab.log({ 
                            'train/train_epoch_loss': epoch_loss/epoch_iter,
                            'train/train_epoch_acc1': epoch_acc1,
                            'train/train_epoch_acc5': epoch_acc5,
                            'val/val_cma_loss': val_log['val_cma_loss'],
                            'val/val_uma_loss': val_log['val_uma_loss'],
                            'val/val_local_loss': val_log['val_local_loss'],
                            'val/val_epoch_loss': val_log['val_loss'],
                            'val/val_epoch_acc1': val_log['val_acc1'],
                            'val/val_epoch_acc5': val_log['val_acc5']}
                            )
                
                
                # zero-shot eval      
                avg_f1, avg_acc, avg_auc = 0, 0, 0
                for set_name in args_zeroshot_eval['val_sets'].keys():

                    f1, acc, auc, _, _, _, res_dict = \
                        zeroshot_eval(model=self.model, 
                        set_name=set_name, 
                        device=self.device, 
                        args_zeroshot_eval=args_zeroshot_eval)

                    avg_f1 += f1
                    avg_acc += acc
                    avg_auc += auc

                    # log each val set zeroshot performance
                    swanlab.log({ 
                                # f'metric/{set_name}_f1': f1,
                                # f'metric/{set_name}_acc': acc,
                                f'metric/{set_name}_AUROC': auc
                                }
                                )
                
                avg_f1 = avg_f1/len(args_zeroshot_eval['val_sets'].keys())
                avg_acc = avg_acc/len(args_zeroshot_eval['val_sets'].keys())
                avg_auc = avg_auc/len(args_zeroshot_eval['val_sets'].keys())
                swanlab.log({
                            'metric/avg_f1': avg_f1,
                            'metric/avg_acc': avg_acc,
                            'metric/avg_auc': avg_auc
                            }
                            )
                

                best_metric = avg_auc
                if best_metric > best_auc:
                    best_auc = best_metric
                    no_improve_count = 0
                    torch.save(self.model.module.state_dict(),
                               os.path.join(model_checkpoints_folder, f"{self.model_name}_bestZeroShotAll_ckpt.pth"))
                    torch.save(self.model.module.ecg_encoder.state_dict(),
                                os.path.join(model_checkpoints_folder, f"{self.model_name}_bestZeroShotAll_encoder.pth"))
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch_counter} "
                              f"(no improvement for {patience} epochs, "
                              f"best AUROC={best_auc:.4f})")
                        break

    def val(self, loader):
        print('start validation')
        self.model.eval()
        val_cma_loss = 0
        val_uma_loss = 0
        val_local_loss = 0
        val_loss = 0
        val_epoch_acc1 = []
        val_epoch_acc5 = []
        
        for data in tqdm(loader):
            # get raw text
            report = data['report']
            # get ecg
            ecg = data['ecg'].to(torch.float32).to(
                self.device).contiguous()
            labels = data.get('labels', None)
            if self.use_jaccard_mask:
                if labels is None:
                    raise KeyError('labels not found in dataset, but use_jaccard_mask is True.')
                if not torch.is_tensor(labels):
                    labels = torch.tensor(labels)
                labels = labels.to(torch.float32).to(self.device).contiguous()
            
            report_tokenize_output = self.model.module._tokenize(report)

            input_ids = report_tokenize_output.input_ids.to(
                self.device).contiguous()
            attention_mask = report_tokenize_output.attention_mask.to(
                self.device).contiguous()
            
            with torch.no_grad():
                output_dict = self.model(
                    ecg,
                    input_ids,
                    attention_mask,
                    return_text_tokens=self.use_local_loss,
                    max_token_len=self.local_max_token_len,
                )
                ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'],\
                                                        output_dict['proj_ecg_emb'],\
                                                        output_dict['proj_text_emb']

                world_size = torch_dist.get_world_size()
                with torch.no_grad():
                    agg_proj_img_emb = [torch.zeros_like(proj_ecg_emb[0]) for _ in range(world_size)]
                    agg_proj_text_emb = [torch.zeros_like(proj_text_emb[0]) for _ in range(world_size)]

                    dist.all_gather(agg_proj_img_emb, proj_ecg_emb[0])
                    dist.all_gather(agg_proj_text_emb, proj_text_emb[0])

                    if self.use_uma_loss:
                        agg_proj_ecg_emb1 = [torch.zeros_like(ecg_emb[0]) for _ in range(world_size)]
                        agg_proj_ecg_emb2 = [torch.zeros_like(ecg_emb[1]) for _ in range(world_size)]
                        dist.all_gather(agg_proj_ecg_emb1, ecg_emb[0])
                        dist.all_gather(agg_proj_ecg_emb2, ecg_emb[1])
                    if self.use_jaccard_mask:
                        agg_labels = [torch.zeros_like(labels) for _ in range(world_size)]
                        dist.all_gather(agg_labels, labels)
                    # get current rank
                    rank = torch_dist.get_rank()

                agg_proj_img_emb[rank] = proj_ecg_emb[0]
                agg_proj_text_emb[rank] = proj_text_emb[0]

                if self.use_uma_loss:
                    agg_proj_ecg_emb1[rank] = ecg_emb[0]
                    agg_proj_ecg_emb2[rank] = ecg_emb[1]
                if self.use_jaccard_mask:
                    agg_labels[rank] = labels

                agg_proj_img_emb = torch.cat(agg_proj_img_emb, dim=0)
                agg_proj_text_emb = torch.cat(agg_proj_text_emb, dim=0)

                if self.use_uma_loss:
                    agg_proj_ecg_emb1 = torch.cat(agg_proj_ecg_emb1, dim=0)
                    agg_proj_ecg_emb2 = torch.cat(agg_proj_ecg_emb2, dim=0)
                if self.use_jaccard_mask:
                    agg_labels = torch.cat(agg_labels, dim=0)

                if self.use_jaccard_mask:
                    cma_loss, acc1, acc5 = clip_loss_jaccard(
                        agg_proj_img_emb,
                        agg_proj_text_emb,
                        agg_labels,
                        jaccard_t=self.jaccard_t,
                        soft_neg_scale=self.soft_neg_scale,
                        device=self.device,
                    )
                    if self.use_uma_loss:
                        uma_loss, _, _ = clip_loss_jaccard(
                            agg_proj_ecg_emb1,
                            agg_proj_ecg_emb2,
                            agg_labels,
                            jaccard_t=self.jaccard_t,
                            soft_neg_scale=self.soft_neg_scale,
                            device=self.device,
                        )
                    else:
                        uma_loss = torch.zeros(1, device=self.device).squeeze()
                else:
                    cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)
                    if self.use_uma_loss:
                        uma_loss, _, _ = clip_loss(agg_proj_ecg_emb1, agg_proj_ecg_emb2, device=self.device)
                    else:
                        uma_loss = torch.zeros(1, device=self.device).squeeze()

                if self.use_local_loss:
                    text_token_emb = output_dict['text_token_emb']
                    text_token_mask = output_dict['text_token_mask']
                    ecg_token_emb = output_dict['ecg_token_emb']
                    local_loss = gloria_local_loss(
                        ecg_token_emb,
                        text_token_emb,
                        text_token_mask,
                        temp1=self.local_temp1,
                        temp2=self.local_temp2,
                        temp3=self.local_temp3,
                    )
                else:
                    local_loss = torch.zeros(1, device=self.device).squeeze()

                loss = cma_loss + uma_loss + self.local_loss_weight * local_loss

                # accumalate loss for logging
                val_cma_loss += cma_loss.item()
                val_uma_loss += uma_loss.item()
                val_local_loss += local_loss.item()
                val_loss += loss.item()
                val_epoch_acc1.append(acc1.item())
                val_epoch_acc5.append(acc5.item())
        
        if self.device == 0:
            val_cma_loss = val_cma_loss/len(val_epoch_acc1)
            val_uma_loss = val_uma_loss/len(val_epoch_acc1)
            val_local_loss = val_local_loss/len(val_epoch_acc1)
            val_loss = val_loss/len(val_epoch_acc1)
            val_epoch_acc1 = np.array(val_epoch_acc1).mean()
            val_epoch_acc5 = np.array(val_epoch_acc5).mean()
            
            val_log = {'val_loss': val_loss,
                        'val_cma_loss': val_cma_loss,
                        'val_uma_loss': val_uma_loss,
                        'val_local_loss': val_local_loss,
                        'val_acc1': val_epoch_acc1,
                        'val_acc5': val_epoch_acc5}
            return val_log
        else:
            return None
        
    def save_checkpoints(self, epoch, PATH):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            PATH)
    
