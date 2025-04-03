# PrefDrive: PrefDrive: Enhancing Autonomous Driving through Preference-Guided Large Language Models

This repository contains the LoRA (Low-Rank Adaptation) checkpoints, datset, training codes for LLaMa2-7B fine-tuned with Direct Preference Optimization (DPO). Our framework, PrefDrive, integrates specific driving preferences into autonomous driving models through large language models, significantly improving performance across multiple metrics including distance maintenance, trajectory smoothness, and traffic rule compliance.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Model Details](#model-details)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [License](#license)

## Overview

Recent advances in Large Language Models (LLMs) have shown promise in autonomous driving, but existing approaches often struggle to align with specific driving behaviors (e.g., maintaining safe distances, smooth acceleration patterns) and operational requirements (e.g., traffic rule compliance, route adherence).

PrefDrive addresses this challenge by:
1. Developing a preference learning framework that combines multimodal perception with natural language understanding
2. Leveraging Direct Preference Optimization (DPO) to fine-tune LLMs efficiently on consumer-grade hardware
3. Training on a comprehensive dataset of 74,040 driving sequences with annotated preferences

Through extensive experiments in the CARLA simulator, we demonstrate that our preference-guided approach significantly improves driving performance, with up to:
- 28.1% reduction in traffic light violations
- 8.5% improvement in route completion
- 63.5% reduction in layout collisions

The model is available on Hugging Face: https://huggingface.co/liyun0607/PrefDrive
The dataset is avaliable on Hugging Face: https://huggingface.co/datasets/liyun0607/PrefDrive

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
  - Alpha (α): 16
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj
- **DPO Configuration**:
  - Beta (β): 0.1
  - Loss Type: Sigmoid

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | LLaMA2-7B |
| Training Strategy | LoRA |
| Learning Rate | 1e-5 |
| Batch Size | 4 |
| Gradient Accumulation Steps | 2 |
| Training Epochs | 3 |
| Maximum Sequence Length | 2,048 |
| Warmup Ratio | 0.1 |
| Max Gradient Norm | 0.3 |

### Performance Results

#### Town 01
| Metric | LMDrive (baseline) | PrefDrive (Ours) | Improvement |
|--------|-------------------|-----------------|-------------|
| Composite Score | 53.00 | 56.12 | +5.9% |
| Route Completion | 59.10 | 64.15 | +8.5% |
| Layout Collisions | 0.73 | 0.27 | -63.5% |
| Traffic Light Violations | 0.22 | 0.16 | -28.1% |

#### Town 04
| Metric | LMDrive (baseline) | PrefDrive (Ours) | Improvement |
|--------|-------------------|-----------------|-------------|
| Composite Score | 60.11 | 65.93 | +9.7% |
| Route Completion | 65.25 | 69.93 | +7.2% |
| Traffic Light Violations | 0.24 | 0.00 | -100.0% |

## Methodology

The PrefDrive methodology for autonomous driving is formulated as:

$\mathcal{L}_{DPO} = -\mathbb{E}_{(s,a_p,a_r)\sim\mathcal{D}}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(a_p|s)}{\pi_{ref}(a_p|s)} - \beta\log\frac{\pi_\theta(a_r|s)}{\pi_{ref}(a_r|s)}\Big)\Big]$

where:
- $\mathcal{D}$ represents our driving preference dataset
- $s$ denotes the current driving scenario description
- $a_p$ represents the preferred (chosen) driving action with its reasoning and resulting waypoint
- $a_r$ represents the rejected driving action with its reasoning and resulting waypoint
- $\pi_\theta$ is the policy model being trained
- $\pi_{ref}$ is the initial reference model
- $\beta$ controls the preference learning sensitivity (set to 0.1)
- $\sigma$ represents the sigmoid function

This formulation explicitly shows how our model learns to favor chosen driving actions over rejected ones while maintaining reasonable deviation from the reference model's behavior.

## Dataset

Our comprehensive dataset consists of 74,040 driving sequences, carefully annotated with driving preferences and driving decisions. Each sequence contains:

- A detailed driving scenario description
- A preferred/chosen driving action with reasoning and waypoint
- A rejected driving action with reasoning and waypoint

The dataset captures various autonomous driving scenarios with emphasis on proper distance maintenance, trajectory smoothness, traffic rule compliance, and route adherence.

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
