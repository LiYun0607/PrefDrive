# PrefDrive: Enhancing Autonomous Driving through Preference-Guided Large Language Models

This repository contains the LoRA (Low-Rank Adaptation) [checkpoints]((https://huggingface.co/liyun0607/PrefDrive)), [dataset]((https://huggingface.co/liyun0607/PrefDrive)), and training code for LLaMa2-7B fine-tuning with [Direct Preference Optimization (DPO) algorithm](https://arxiv.org/abs/2305.18290). Our framework, PrefDrive, integrates driving preferences into autonomous driving models through large language models, significantly improving performance across multiple metrics including distance maintenance, trajectory smoothness, and traffic rule compliance.

## Table of Contents

- [Overview](#overview)
- [Model Details](#model-details)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Performance Results](#performance-results)
- [License](#license)
- [Citation](#citation)

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

## Methodology

![PrefDrive Architecture](https://github.com/LiYun0607/PrefDrive/blob/main/assets/DPO-system-structure.png)

This formulation explicitly shows how our model learns to favor chosen driving actions over rejected ones while maintaining reasonable deviation from the reference model's behavior.

## Dataset

Our comprehensive dataset consists of 74,040 driving sequences, carefully annotated with driving preferences and driving decisions. Each sequence contains:

- A detailed driving scenario description
- A preferred/chosen driving action with reasoning and waypoint
- A rejected driving action with reasoning and waypoint

The dataset captures various autonomous driving scenarios with emphasis on proper distance maintenance, trajectory smoothness, traffic rule compliance, and route adherence.

## Performance Results

### Comparative Analysis in CARLA Town 01 and Town 04

| Method | Composite Score (↑) | Penalty Score (↑) | Route Completion (↑) | Layout Collisions (↓) | Traffic Light Violations (↓) | Route Deviation (↓) | Vehicle Blocked (↓) |
|--------|---------------------|-------------------|----------------------|-----------------------|------------------------------|---------------------|---------------------|
| **Town 01** |
| LMDrive (baseline) | 53.00 | 0.86 | 59.10 | 0.73 | 0.22 | **1.32** | 0.11 |
| PrefDrive (ours) | **56.12** (+5.9%) | 0.88 (+1.5%) | **64.15** (+8.5%) | **0.27** (-63.5%) | **0.16** (-28.1%) | 1.36 (+3.0%) | **0.00** (-100.0%) |
| NoPrefDPO | 51.45 (-2.9%) | **0.91** (+4.8%) | 55.17 (-6.7%) | 0.61 (-17.0%) | 0.20 (-11.9%) | 1.74 (+31.7%) | 0.08 (-25.0%) |
| **Town 04** |
| LMDrive (baseline) | 60.11 | 0.93 | 65.25 | 0.00 | 0.24 | 1.86 | 0.00 |
| PrefDrive (ours) | **65.93** (+9.7%) | **0.96** (+3.2%) | **69.93** (+7.2%) | 0.00 (0.0%) | **0.00** (-100.0%) | **1.77** (-4.8%) | 0.00 (0.0%) |
| NoPrefDPO | 62.27 (+3.6%) | 0.94 (+0.7%) | 68.35 (+4.7%) | 0.00 (0.0%) | 0.05 (-80.1%) | 1.77 (-4.6%) | 0.00 (0.0%) |

## License

[Apache License 2.0](LICENSE)

## Citation

If you use this model or dataset in your research, please cite:

```bibtex
@article{li2025prefdrive,
      title={PrefDrive: A Preference Learning Framework for Autonomous Driving with Large Language Models}, 
      author={Li, Yun and Javanmardi, Ehsan and Thompson, Simon and Katsumata, Kai and Orsholits, Alex and Tsukada, Manabu},
      year={2025},
      journal={IEEE Intelligent Vehicles Symposium (IV)},
}
```
