import os

from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoConfig, AutoModelForCausalLM,LlavaProcessor
from .conversation import conv_templates
from .model.language_model.llava_qwen2 import LlavaQwenConfig,LlavaQwenConfig, LlavaQwen2ForCausalLM
from .model.language_model.llava_llama import LlavaLlamaModel, LlavaLlamaForCausalLM,LlavaConfig
from .model.constants import DEFAULT_IMAGE_TOKEN
from .model.utils import download

# AutoConfig.register("llava", LlavaQwenConfig)
AutoConfig.register("llava_qwen2", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwen2ForCausalLM)
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


class HuatuoGPT:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model= model_path,
            enforce_eager=True,
            tensor_parallel_size= int(os.environ.get("tensor_parallel_size",1)),
            # limit_mm_per_prompt = {"image": int(os.environ.get("max_image_num",1))},
        )

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )


    def process_messages(self,messages):
        conv = conv_templates["huatuo"].copy()
        conv.messages = []
        if "system" in messages:
            conv.system = "<|system|>" + messages["system"]
        else:
            conv.system = ""
        
        imgs = []
        if "image" in messages:
            text = messages["prompt"]
            inp = DEFAULT_IMAGE_TOKEN + '\n' + text
            conv.append_message(conv.roles[0],inp)
            image = messages["image"]
            if isinstance(image,str):
                if os.path.exists(image):
                    image = Image.open(image)
                else:
                    image = download(image)
            imgs.append(image)
        elif "images" in messages:
            text = messages["prompt"]
            images = messages["images"]
            inp = ""
            for i,image in enumerate(images):
                inp = inp + f"<image_{i+1}>: " + DEFAULT_IMAGE_TOKEN + '\n'
                if isinstance(image,str):
                    if os.path.exists(image):
                        image = Image.open(image)
                    else:
                        image = download(image)
                elif isinstance(image,Image.Image):
                    image = image.convert("RGB")
                imgs.append(image)
            inp = inp + text
            conv.append_message(conv.roles[0],inp)
        else:
            text = messages["prompt"]
            inp = text
            conv.append_message(conv.roles[0],inp)
        
        conv.append_message(conv.roles[1],None) 
        prompt = conv.get_prompt()

        mm_data = {}
        # print(len(imgs))
        if len(imgs) > 0:
            mm_data["image"] = imgs


        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        return llm_inputs

    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_outputs(self,messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        # from pdb import set_trace;set_trace()
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res

