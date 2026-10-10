import torch
import os
import json
import gc
import csv
import numpy as np
from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from ..question_formats import get_report_generation_prompt

from ..utils import save_json, extract
from ..base_dataset import BaseDataset


class CTRATE(BaseDataset):
    def __init__(self, model, dataset_path, output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path 
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))

    def run(self, samples, model):
        out_samples = []  # Add this line to initialize the list
        with torch.no_grad():
            messages_list = []
            current_messages = []
            current_samples = []
            for sample in tqdm(samples, desc="Batching samples"):
                messages = sample["messages"]
                current_messages.append(messages)
                current_samples.append(sample)
                if len(current_messages) >= 50: 
                    messages_list.append([current_messages, current_samples])
                    current_messages = []
                    current_samples = []
            if current_messages:
                messages_list.append([current_messages, current_samples])
            
            for current_messages, current_samples in tqdm(messages_list, desc="Generating reports"):
                outputs = model.generate_outputs(current_messages)
                try:
                    for sample, response in zip(current_samples, outputs):
                        del sample["messages"]
                        sample["response"] = response
                        out_samples.append(sample)  # Now this will work
                except Exception as e:
                    print(f"Error processing batch: {e}")
                gc.collect()
        return out_samples

    def load_data(self):
        json_path = ("/path/to/your/model/CT-RATE/validation_reports_output_add_image.json") 
        
        print(f"Loading AMOS data from: {json_path}")
        
        all_lines = []
        with open(json_path,"r") as f:
            all_lines = json.load(f)

        chunked_lines = np.array_split(all_lines, self.num_chunks)[self.chunk_idx].tolist()

        for sample in tqdm(chunked_lines, desc=f"Processing chunk {self.chunk_idx}"):
            sample = self.construct_messages(sample)
            if sample: 
                self.samples.append(sample)
                
        print(f"Chunk {self.chunk_idx}: Loaded {len(self.samples)} samples.")
        return self.samples

    def construct_messages(self, sample):

        try:
            prompt = get_report_generation_prompt()
            prompt = prompt.replace("<video>\n", "").strip()
        except (KeyError, IndexError):
            print(f"Warning: Could not find prompt in sample. Skipping. Data: {sample}")
            return None

        video_dir = sample.get("video")[0]
        if not video_dir or not os.path.isdir(video_dir):
            print(f"Warning: Video directory not found or is not a directory. Skipping. Path: {video_dir}")
            return None
        
        image_files = sorted([
            f for f in os.listdir(video_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.dcm'))
        ])
        
        if not image_files:
            print(f"Warning: No image files found in directory. Skipping. Path: {video_dir}")
            return None

        images = []
        for f in image_files:
            with open(os.path.join(video_dir, f), 'rb') as file:
                img = Image.open(file)
                images.append(img.copy())
                img.close()  


        messages = {"prompt": prompt, "images": images}
        sample["messages"] = messages
        
        return sample

    def cal_metrics(self, out_samples):

        import pandas as pd

        predictions_data = []
        ground_truth_data = []

        for i, sample in enumerate(out_samples):
            response = sample["response"]
            golden = sample['conversations'][1]['value']

            study_id = f"study_{self.chunk_idx}_{i+1}" 
            
            predictions_data.append({'study_id': study_id, 'report': response})
            ground_truth_data.append({'study_id': study_id, 'report': golden})

        predictions_df = pd.DataFrame(predictions_data)
        ground_truth_df = pd.DataFrame(ground_truth_data)

        os.makedirs(self.output_path, exist_ok=True)
        
        prediction_path = os.path.join(self.output_path, f'predictions_{self.chunk_idx}.csv')
        ground_truth_path = os.path.join(self.output_path, f'ground_truth_{self.chunk_idx}.csv')
        predictions_df.to_csv(prediction_path, index=False)
        ground_truth_df.to_csv(ground_truth_path, index=False)

        print(f"Chunk {self.chunk_idx} results saved to {self.output_path}")

        return {"total metrics": "please use cal_report_metrics.py to generate metrics"}, out_samples
