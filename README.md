# DyMoMed

# workflow文件夹中为流程运行的代码，实际运行可参考readme_workflow.md，交互流程运行run_inter.sh，评估运行run_eval.sh.

# data文件夹中为数据处理的代码和使用的数据.其中clinic100.json\mtmed100.json\pmc100.json为我们处理好的数据.


# DyMoMed: A Multi-Path Reasoning Agent with Dynamic Goal Adaptation

This repository contains the code and data for the paper: **"DyMoMed: A Multi-Path Reasoning Agent with Dynamic Goal Adaptation for Multi-Objective Medical Dialogue under Imperfect Patient-Reported Information."**

## 📂 Code Structure & Usage

The core logic for the agent's workflow is located in the `workflow/` directory.

* **Detailed Instructions:** Please refer to `readme_workflow.md` for a comprehensive guide on running the code.
* **Interactive Mode:** To run the interactive dialogue process, execute the following script:
```bash
bash run_inter.sh

```


* **Evaluation:** To run the evaluation metrics, execute:
```bash
bash run_eval.sh

```

## 💾 Datasets

The `data/` directory contains data processing scripts and the datasets used for experimentation.

We provide the following pre-processed datasets:

* `clinic100.json`
* `mtmed100.json`
* `pmc100.json`
