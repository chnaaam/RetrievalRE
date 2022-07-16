import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import torch
import numpy as np
import argparse
from trainer import Trainer
from builder import Builder
from evaluator import Evaluator

from file_io import *


def fix_seed(random_seed=42):
    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    np.random.seed(random_seed)
    random.seed(random_seed)
    
    
def init_dirs(args):
    make_dir(args.cache_path)
    make_dir(args.model_path)

    
if __name__ == "__main__":
    fix_seed()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--run_mode", default="evaluate")
    
    parser.add_argument("--plm", default="klue/roberta-base")
    parser.add_argument("--pretrained_model", default="/home/chnaaam/chnaaam/KnowPrompt/pretrained_model")
        
    parser.add_argument("--data_path", default="/home/chnaaam/data/KLUE/klue_benchmark/klue-re-v1.1/")
    parser.add_argument("--train_data_fn", default="klue-re-v1.1_train.json")
    parser.add_argument("--valid_data_fn", default="klue-re-v1.1_dev.json")
    
    parser.add_argument("--cache_path", default="./cache")
    parser.add_argument("--model_path", default="./model")
    
    parser.add_argument("--use_gpu", default=True)
    parser.add_argument("--use_fp16", default=False)
    parser.add_argument("--max_seq_length", default=256)
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--lr", default=3e-5)
    parser.add_argument("--batch_size", default=20)
    parser.add_argument("--train_num_workers", default=0)
    parser.add_argument("--valid_num_workers", default=0)
    
    parser.add_argument("--topk", default=12)
    parser.add_argument("--logit_ratio", default=0.4)
    
    args = parser.parse_args()
    
    init_dirs(args)
    
    if args.run_mode == "train":
        trainer = Trainer(args=args)
        trainer.run()
        
    elif args.run_mode == "build":
        builder = Builder(args=args)
        builder.run()
        
    elif args.run_mode == "evaluate":
        evaluator = Evaluator(args=args)
        evaluator.run()