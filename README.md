# LoRA DPO LLaMa-7B

This repository contains the LoRA (Low-Rank Adaptation) parameters for LLaMa-7B fine-tuned with Direct Preference Optimization (DPO). These parameters can be used to enhance the base model's capabilities by better aligning it with human preferences.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Model Details](#model-details)
- [License](#license)

## Overview

This LoRA adapter was trained using Direct Preference Optimization (DPO) on pairwise preference data. DPO allows for directly aligning language models with human preferences without requiring a separate reward model, as is typical in RLHF approaches.

The model is available on Hugging Face: [YOUR_USERNAME/lora-dpo-llama-7b](https://huggingface.co/YOUR_USERNAME/lora-dpo-llama-7b)

## Installation

```bash
pip install transformers peft torch unsloth
```

## Usage

### Basic Usage with Transformers

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model_id = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id)

# Load LoRA adapter
peft_model_id = "YOUR_USERNAME/lora-dpo-llama-7b"
model = PeftModel.from_pretrained(model, peft_model_id)

# Use model for inference
inputs = tokenizer("Hello, please", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Faster Inference with Unsloth

```python
from unsloth import FastLanguageModel

# Load the model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_4bit=True,
    max_seq_length=2048
)

# Load LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    "YOUR_USERNAME/lora-dpo-llama-7b",
)

# Use model for inference
inputs = tokenizer("Hello, please", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

To reproduce the training or train your own version with your dataset:

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/lora-dpo-llama.git
cd lora-dpo-llama
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python train_dpo.py \
  --model_path "/path/to/llama-7b-base" \
  --output_path "/path/to/output/directory" \
  --json_path "/path/to/your/dataset.json" \
  --num_epochs 3 \
  --learning_rate 1e-5 \
  --loss_type "exo_pair" \
  --beta 0.1
```

For more options:
```bash
python train_dpo.py --help
```

## Model Details

- **Base Model**: meta-llama/Llama-2-7b
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 16
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj
- **DPO Configuration**:
  - Beta: 0.1
  - Loss Type: exo_pair

## License

[Apache License 2.0](LICENSE)

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{rafailov2023direct,
      title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model}, 
      author={Rafael Rafailov and Archit Sharma and Eric Mitchell and Stefano Ermon and Christopher D. Manning and Chelsea Finn},
      year={2023},
      eprint={2305.18290},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{hu2021lora,
      title={LoRA: Low-Rank Adaptation of Large Language Models}, 
      author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
      year={2021},
      eprint={2106.09685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
