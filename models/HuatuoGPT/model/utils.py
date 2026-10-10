from transformers import AutoConfig
import requests

def download(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/87.0.4280.88 Safari/537.36'
        }  # Mimic a common browser's User-Agent
        response = requests.get(url,headers=headers,timeout=120)
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to download the file from the URL: {url}")
            return None
    except requests.RequestException as e:
        print(f"An error occurred while downloading the file from the URL: {url}")
        print(e)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while downloading the file from the URL: {url}")
        print(e)
        return None

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
