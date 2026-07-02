# MRGEvalKit
MRGEvalKit is the evaluation framework for AtomiMed, a comprehensive multi-modal benchmark spanning X-ray, CT, MRI, and Ultrasound modalities. Our detailed repository will be coming soon.

## Installation
To install requirements:
`pip install -r requirements_qwen3.txt`


## Evaluation
### 1. set variables
| Variables | Supported Options |
| :--- | :--- |
| `EVAL_DATASETS` | - `IU_XRAY`<br>- `MIMIC_CXR`<br>- `CheXpert_Plus`<br>- `Amos`<br>- `CT-RATE`<br>- `Merlin`<br>- `RadGenome_Brain_MRI`<br>- `Ultrasound`<br>|
| `MODEL_NAME` | - `HuatuoGPT`<br>- `HuluMed_Qwen2`<br>- `HuluMed_Qwen3`<br>- `InternVL`<br>- `MedGemma`<br>- `Qwen2_5_VL`<br>- `Qwen3_VL`<br>|

set `EVAL_DATASETS`, `MODEL_NAME`, `MODEL_PATH`, `OPENAI_API_KEY` in `eval_chunked.sh`, for example:
```
EVAL_DATASETS="RadGenome_Brain_MRI"
MODEL_NAME="MedGemma"
MODEL_PATH="/path/to/your/ckpt/medgemma-27b-it"
OPENAI_API_KEY="Your key"
```

### 2. run the script
```
cd /path/to/your/root/MRGEvalKit
bash eval_chunked.sh
```

