import os
import torch
import faiss
import pickle as pkl


class OpenBookDataStore:
    def __init__(self, plm_hidden_size):
        self.faiss_index = faiss.IndexFlatL2(plm_hidden_size)
        
        self.hidden_states = []
        self.index_relation_pairs = {}
        
    def add(self, hidden_state, labels):
        self.hidden_states.append(hidden_state)
        
        len_index_relation_pairs = len(self.index_relation_pairs)
        
        for idx, label in enumerate(labels):
            self.index_relation_pairs.setdefault(
                len_index_relation_pairs + idx,
                label
            )
        
    def build(self, model_path):
        self.faiss_index.add(torch.cat(self.hidden_states).numpy())
        
        self.save_index(
            index=self.faiss_index,
            dump_path=os.path.join(model_path, "faiss_index.dump")
        )
        
        self.save_index_relation_pairs(
            index_relation_pairs=self.index_relation_pairs,
            dump_path=os.path.join(model_path, "index_relation_pairs.dump")
        )
        
    def load(self, model_path):
        self.faiss_index = self.load_index(dump_path=os.path.join(model_path, "faiss_index.dump"))
        self.index_relation_pairs = self.load_index_relation_pairs(dump_path=os.path.join(model_path, "index_relation_pairs.dump"))
    
    def search(self, hidden_state, topk):
        return self.faiss_index.search(hidden_state, topk)
    
    def get_label_from_index(self, index):
        return self.index_relation_pairs[index]
    
    @staticmethod
    def save_index(index, dump_path):
        faiss.write_index(index, dump_path)
    
    @staticmethod
    def load_index(dump_path):
        return faiss.read_index(dump_path)
    
    @staticmethod
    def save_index_relation_pairs(index_relation_pairs, dump_path):
        with open(dump_path, "wb") as fp:
            pkl.dump(index_relation_pairs, fp)
            
    @staticmethod
    def load_index_relation_pairs(dump_path):
        with open(dump_path, "rb") as fp:
            return pkl.load(fp)