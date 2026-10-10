from tqdm import tqdm
import os
from .cli import HuatuoChatbot


class HuatuoGPT:
    def __init__(self,model_path,args):
        super().__init__()
        self.bot = HuatuoChatbot(model_path)
        
        self.bot.gen_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'min_new_tokens': 1,
            'repetition_penalty': args.repetition_penalty,
            'temperature' : args.temperature,
            'top_p' : args.top_p
        }

    def process_messages(self,messages):
        question = ""
        images = []
        if "system" in messages:
            question =  messages["system"]
        question += messages["prompt"]
        if "image" in messages:
            images = messages["image"]
        elif "images" in messages:
            images = messages["images"]
        
        llm_inputs = {
            "question": question,
            "images": images,
        }
        return llm_inputs

    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        query = llm_inputs["question"]
        images = llm_inputs["images"]
        outputs = self.bot.inference(query, images)
        print(outputs)
        return outputs[0]
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in tqdm(messages_list,total=len(messages_list)):
            result = self.generate_output(messages)
            res.append(result)
        return res

