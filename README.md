# AnomalyCoT

## Introduction

This repository is the relevant fine-tuning benchmark experiment of the paper PoetryCoT: A Multi-Scenario Chain-of-Thought Dataset for Multimodal Large Language Models

The related experiments of this dataset are based on the LLama-factory framework:https://github.com/hiyouga/LLaMA-Factory

## Getting Started

### Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
git clonehttps://github.com/Marine318/PoetryCoT.git
rm -f data && mv /PoetryCoT/* .
```

### Run Our Script

By running the script, fine-tuning experiments of the relevant model can be conducted. The pre-training weights of the relevant model can be placed in the ```Model_pre``` folder

```bash
llamafactory-cli PoetryCoT_scripts/qwen2_vl_7b_lora.yaml
```

**Our script was originally based on our path. Remember to correct it.**

### Evaluation

We provided the evaluation code for the checkpoints of the relevant fine-tuned model.

```bash
python PoetryCoT_eval/eval_qwen2.5_vl_7b.py
```

After obtaining the relevant reasoning output, we also provided the script for calculating the relevant indicators.

```bash
python evaluate_results_new.py
```

## Acknowledgement

This repo benefits from [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory).