import os


class TestModel:
    def __init__(self,model_path = "/mnt/workspace/workgroup_dev/longli/models/hub/Qwen2.5-VL-7B-Instruct",temperature=0,top_p=0.001,repetition_penalty=1):
        super().__init__()

    def generate_output(self,messages):
        return "A"
    
    def generate_outputs(self,messages_list):
        return ["A" for _ in messages_list]
