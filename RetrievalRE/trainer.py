import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoModelForMaskedLM, AutoTokenizer
from accelerate import Accelerator
from sklearn.metrics import f1_score

from dataset import KlueDataset
from special_tokens import SPECIAL_ENTITY_MARKERS, get_relation_labels


class Trainer:
    def __init__(self, args):
        self.args = args
        
        # Get relations in KLUE training dataset
        relations = KlueDataset.load_relations(
            data_path=args.data_path,
            data_fn=args.train_data_fn,
            cache_path=args.cache_path,
        )
        
        # Verbalization
        relation_labels = get_relation_labels(num_labels=len(relations))
        self.relation_label_map = {r: l for r, l in zip(relations, relation_labels)}
             
        # Load tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm)
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": SPECIAL_ENTITY_MARKERS + relation_labels
        })
        
        # Save tokenizer
        self.tokenizer.save_pretrained(
            os.path.join(
                self.args.model_path, 
                f"{self.args.plm.replace('/', '_')}-tokenizer"
            )
        )
        
        # Load model and resize token embeddings
        self.model = AutoModelForMaskedLM.from_pretrained(args.plm)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Load data loader
        self.train_data_loader = DataLoader(
            dataset=KlueDataset(
                tokenizer=self.tokenizer,
                data_path=args.data_path,
                data_fn=args.train_data_fn,
                cache_path=args.cache_path,
                cache_fn=f"{self.args.plm.replace('/', '_')}.cache.train",
                relation_label_map=self.relation_label_map,
                max_seq_length=args.max_seq_length,
                special_entity_markers=SPECIAL_ENTITY_MARKERS
            ),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.train_num_workers
        )
        
        self.valid_data_loader = DataLoader(
            dataset=KlueDataset(
                tokenizer=self.tokenizer,
                data_path=args.data_path,
                data_fn=args.valid_data_fn,
                cache_path=args.cache_path,
                cache_fn=f"{self.args.plm.replace('/', '_')}.cache.valid",
                relation_label_map=self.relation_label_map,
                max_seq_length=args.max_seq_length,
                special_entity_markers=SPECIAL_ENTITY_MARKERS
            ),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.train_num_workers
        )
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=float(args.lr))
        self.criterion = nn.CrossEntropyLoss()
        
        self.accelerator = Accelerator(
            cpu=False if self.args.use_gpu else True,
            fp16=args.use_fp16
        )
        
        self.model, self.optimizer, self.train_data_loader = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.train_data_loader
        )
        
        self.device = self.accelerator.device
        
    def run(self):
        avg_train_loss, avg_valid_loss, avg_valid_score = 0.0, 0.0, 0.0
        
        for epoch in range(self.args.epochs):
            avg_train_loss = self.train(epoch)
            avg_valid_loss, avg_valid_score = self.valid(epoch)
            
            # Save trained model
            self.model.save_pretrained(
                os.path.join(
                    self.args.model_path,
                    f"e-{epoch}.plm-{self.args.plm.replace('/', '_')}.train-loss-{avg_train_loss:.4f}.valid-loss-{avg_valid_loss:.4f}.score-{avg_valid_score:.2f}"
                )
            )
        
    def train(self, epoch):
        self.model.train()
        
        losses = []
        avg_loss = 0.0
        
        progress_bar = tqdm(self.train_data_loader)
        for batch in progress_bar:
            progress_bar.set_description(f"[Training] Epoch : {epoch}, Avg Loss : {avg_loss:.4f}")
            
            inputs, labels = batch
            
            logits = self.model(**inputs).logits
            
            mask_logits = self.get_mask_logits(
                logits=logits,
                mask_idxes=(inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            )
            
            loss = self.criterion(mask_logits, labels)
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            
            losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            
        return sum(losses) / len(losses)
    
    def valid(self, epoch):
        self.model.eval()
        
        losses, scores = [], []
        avg_loss, avg_score = 0.0, 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(self.valid_data_loader)
            for batch in progress_bar:
                progress_bar.set_description(f"[Validation] Epoch : {epoch}, Avg Loss : {avg_loss:.4f}, Avg F1 Score : {avg_score:.2f}")
                inputs, labels = batch
            
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                logits = self.model(**inputs).logits
                
                mask_logits = self.get_mask_logits(
                    logits=logits,
                    mask_idxes=(inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                )
                
                loss = self.criterion(mask_logits, labels)
                losses.append(loss.item())
                
                f1_score = self.calc_f1_score(
                    true_y=labels.tolist(), 
                    pred_y=torch.argmax(torch.softmax(mask_logits, dim=-1), dim=-1).tolist()
                )
                scores.append(f1_score)
                
                avg_loss = sum(losses) / len(losses)
                avg_score = (sum(scores) / len(scores)) * 100
            
        return avg_loss, avg_score
    
    def get_mask_logits(self, logits, mask_idxes):
        return logits[torch.arange(logits.shape[0]), mask_idxes]
    
    @staticmethod
    def calc_f1_score(true_y, pred_y):
        return f1_score(true_y, pred_y, average="micro")
    