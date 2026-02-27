from utils import (
    IU_XRAY,
    CheXpert_Plus,
    MIMIC_CXR,
    Amos,
    CTRATE,
    Merlin,
    Ultrasound,
    RadGenome_Brain_MRI
)

def prepare_benchmark(model,eval_dataset,eval_dataset_path,eval_output_path):
    supported_dataset = ["IU_XRAY", "CheXpert_Plus", "MIMIC_CXR"]
    if eval_dataset == "CT-RATE":
        dataset = CTRATE(model,eval_dataset_path,eval_output_path)
    elif eval_dataset == "RadGenome_Brain_MRI":
        dataset = RadGenome_Brain_MRI(model,eval_dataset_path,eval_output_path)
    elif eval_dataset == "Amos":
        dataset = Amos(model,eval_dataset_path,eval_output_path)
    elif eval_dataset == "Merlin":
        dataset = Merlin(model,eval_dataset_path,eval_output_path)
    elif eval_dataset == "IU_XRAY":
        dataset = IU_XRAY(model,eval_dataset_path,eval_output_path)
    elif eval_dataset == "CheXpert_Plus":
        dataset = CheXpert_Plus(model,eval_dataset_path,eval_output_path)
    elif eval_dataset == "MIMIC_CXR":
        dataset = MIMIC_CXR(model,eval_dataset_path,eval_output_path)
    elif eval_dataset == "Ultrasound":
        dataset = Ultrasound(model,eval_dataset_path,eval_output_path)
    else:
        print(f"unknown eval dataset {eval_dataset}, we only support {supported_dataset}")
        dataset = None

    return dataset

if __name__ == '__main__':
    pass    