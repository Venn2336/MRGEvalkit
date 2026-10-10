import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

import numpy as np

from ..utils import save_json,extract
from ..base_dataset import BaseDataset

from ..question_formats import get_report_generation_prompt

class CheXpert_Plus(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        dataset_path = self.dataset_path
        json_path = "./chexpert_val.json"

        with open(json_path,"r") as f:
            dataset = json.load(f)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages(self,sample):
        image = sample["image"][0]
        image = Image.open(image)
        golden = sample['conversations'][1]['value']
        
        prompt = get_report_generation_prompt()
        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages
        return sample


    def cal_metrics(self,out_samples):
        import pandas as pd

        predictions_data = []
        ground_truth_data = []

        for i,sample in enumerate(out_samples):
            response = sample["response"]
            golden = sample['conversations'][1]['value']

            study_id = f"study_{i+1}"
            
            predictions_data.append({
                'study_id': study_id,
                'report': response
            })

            ground_truth_data.append({
                'study_id': study_id,
                'report': golden
            })


        predictions_df = pd.DataFrame(predictions_data)
        ground_truth_df = pd.DataFrame(ground_truth_data)

        prediction_path = os.path.join(self.output_path,'predictions.csv')
        ground_truth_path = os.path.join(self.output_path,'ground_truth.csv')
        predictions_df.to_csv(prediction_path, index=False)
        ground_truth_df.to_csv(ground_truth_path, index=False)

        return {"total metrics":"please use cal_report_metrics.py to generate metrics"},out_samples
                
    def maybe_download_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist. Please download from https://aimi.stanford.edu/datasets/chexpert-plus and put it under {self.dataset_path}.")
                