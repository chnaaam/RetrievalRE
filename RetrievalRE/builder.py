from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForMaskedLM, AutoTokenizer

from dataset import KlueDataset
from special_tokens import SPECIAL_ENTITY_MARKERS, get_relation_labels
from open_book_data_store import OpenBookDataStore


class Builder:
    def __init__(self, args):
        self.args = args
        self.device = "cuda:0" if args.use_gpu else "cpu"
        
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
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
                
        # Load model and resize token embeddings
        self.model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
        self.model = self.model.to(self.device)
        
        # Load open book data store
        self.data_store = OpenBookDataStore(self.model.config.hidden_size)
        
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
        
    def run(self):
        # Build open book data store
        self.build()
        
    def build(self):
        self.model.eval()
        
        with torch.no_grad():
            progress_bar = tqdm(self.train_data_loader)
            for batch in progress_bar:
                progress_bar.set_description(f"Build open-book data store")
                
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                hidden_state = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
                
                mask_hidden_state = self.get_mask_hidden_state(
                    hidden_state=hidden_state,
                    mask_idxes=(inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                ).cpu()
                
                self.data_store.add(mask_hidden_state, labels.tolist())
                
        self.data_store.build(model_path=self.args.model_path)
    
    def get_mask_hidden_state(self, hidden_state, mask_idxes):
        return hidden_state[torch.arange(hidden_state.shape[0]), mask_idxes]
    