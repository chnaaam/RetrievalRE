import os
import json
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import Dataset


class KlueDataset(Dataset):
    
    def __init__(self, tokenizer, data_path, data_fn, cache_path, cache_fn, relation_label_map, max_seq_length=256, special_entity_markers=["[E1]", "[/E1]", "[E2]", "[/E2]"]):
        self.data = []
        self.tokenizer = tokenizer
        self.relation_label_map = relation_label_map
        self.max_seq_length = max_seq_length
        self.special_entity_markers = special_entity_markers
        
        cache_full_path = os.path.join(cache_path, cache_fn)
        
        if not os.path.isfile(cache_full_path):
            for data in tqdm(self.read(path=data_path, fn=data_fn), desc="Load KLUE dataset"):
                # Sample Data
                # {
                #     'guid': 'klue-re-v1_train_00000', 
                #     'sentence': '〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.', 
                #     'subject_entity': {
                #         'word': '비틀즈', 
                #         'start_idx': 24, 
                #         'end_idx': 26, 
                #         'type': 'ORG'
                #     }, 
                #     'object_entity': {
                #         'word': '조지 해리슨', 
                #         'start_idx': 13, 
                #         'end_idx': 18, 
                #         'type': 'PER'
                #     }, 
                #     'label': 'no_relation', 
                #     'source': 'wikipedia'
                # }
                
                # Prompt Template Format
                # - [CLS] Steve Jobs, co-founder of Apple.[SEP] Apple [MASK] Steve Jobs [SEP]
                PROMPT_TEMPLATE = "[X] [SEP] [SUB_MARKER] [SUBJECT] [/SUB_MARKER] [MASK] [OBJ_MARKER] [OBJECT] [/OBJ_MARKER]"
                for origin_str, replace_str in [
                    ("[X]", data['sentence']),
                    ("[SUB_MARKER]", self.special_entity_markers[0]),
                    ("[SUBJECT]", data['subject_entity']['word']),
                    ("[/SUB_MARKER]", self.special_entity_markers[1]),
                    ("[OBJ_MARKER]", self.special_entity_markers[2]),
                    ("[OBJECT]", data['object_entity']['word']),
                    ("[/OBJ_MARKER]", self.special_entity_markers[3]),
                ]:
                    PROMPT_TEMPLATE = PROMPT_TEMPLATE.replace(origin_str, replace_str)
                
                # Experiments show that the maximum length of a token created with a prompt template is 236, 
                # so no adjustment is required for 236 and above.   
                input_tensors = self.tokenizer(
                    PROMPT_TEMPLATE, 
                    max_length=self.max_seq_length, 
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                label_id = self.tokenizer(
                    self.relation_label_map[data["label"]],
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"][0]
                
                self.data.append({
                    "inputs": {
                        "input_ids": input_tensors['input_ids'].squeeze(dim=0),
                        "token_type_ids": input_tensors['token_type_ids'].squeeze(dim=0),
                        "attention_mask": input_tensors['attention_mask'].squeeze(dim=0)
                    },
                    "subject_entity_info": data["subject_entity"],
                    "object_entity_info": data["object_entity"],
                    "label": label_id.squeeze(dim=0)
                })
            
            self.save_cache(
                path=cache_full_path,
                data=self.data
            )
        else:
            self.data = self.load_cache(path=cache_full_path)
             
    @staticmethod
    def read(path, fn):
        with open(os.path.join(path, fn), "r") as fp:
            return json.load(fp)
    
    @classmethod
    def load_relations(cls, data_path, data_fn, cache_path):
        cache_full_path = os.path.join(cache_path, "relations.txt")
        
        if not os.path.isfile(cache_full_path):
            relations = cls.get_relations(data_path=data_path, data_fn=data_fn)
            
            with open(cache_full_path, "w", encoding="utf-8") as fp:
                for r in relations:
                    fp.write(f"{r}\n")
        else:
            relations = []
            
            with open(cache_full_path, "r", encoding="utf-8") as fp:
                for line in fp.readlines():
                    line = line.replace("\n", "")
                    
                    if line:
                        relations.append(line)
                    
        return relations
    
    @classmethod
    def get_relations(cls, data_path, data_fn):
        labels = []
        
        for data in tqdm(cls.read(path=data_path, fn=data_fn), desc="Load KLUE dataset to get relations"):
            labels.append(data["label"])
        
        return list(set(labels))
    
    @staticmethod
    def save_cache(path, data):
        with open(path, "wb") as fp:
            pkl.dump({"data": data}, fp)
    
    @staticmethod
    def load_cache(path):
        with open(path, "rb") as fp:
            return pkl.load(fp)["data"]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        
        return data["inputs"], data["label"]