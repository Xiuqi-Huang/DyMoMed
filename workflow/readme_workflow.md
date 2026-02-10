## 📂 File Descriptions

### Model & Agent Modules

* **`base_doctor.py`**: Implements the baseline doctor agent models used for comparative experiments.
* **`base_doctor_dagent.py`**: Implements the **DoctorAgent-RL** model.
* **`patient.py`**: Contains the models for both the **Patient Agent** and the **Patient Monitor Agent**.
* **`utils.py`**: A collection of fundamental utility functions.

### Workflow Scripts

* **`interaction.py`**: Handles the core doctor-patient interaction loop.
* **`evaluation.py`**: Performs evaluation metrics on the doctor agent's output.

---

## ⚙️ Configuration & Execution Guide

Before running the scripts, please ensure the following paths and parameters are configured to match your local environment.

### 1. Script Configuration (`workflow/`)

#### `interaction.py`

* **Path Configuration**: Verify that `result_dir` matches your local directory structure:
```python
result_dir = f"../result/inter/{args.agent_dataset}"

```


* **API Configuration**: Fill in the default values for your API base URLs in the `apiparser`:
* `--doctor_base_url` (Default: `'to be filled'`)
* `--patient_base_url` (Default: `'to be filled'`)



#### `evaluation.py`

* **Path Configuration**: Verify the output directory path:
```python
out_dir = f"../result/eval/{args.dataset}"

```



#### `utils.py`

* **Data Mapping**: Ensure the `DATA_MAP` dictionary points to the correct relative paths for your datasets:
```python
DATA_MAP = {
    'pmc': "../data/pmc100.json",
    'clinic': "../data/clinic100.json",
    'mtmed': "../data/mtmed100.json"
}

```



### 2. Execution Scripts (`bash`)

#### `run_inter.sh` (Interaction)

Configure the following parameters before running:

* `KEY`: Your API key ("to be filled").
* `PYTHON_SCRIPT`: Path to the interaction script (e.g., `..\workflow\interaction.py`).
* `DOCTOR_MODEL`: The full API model name (e.g., `google/gemini-3-flash-preview`).

#### `run_eval.sh` (Evaluation)

Configure the following parameters before running:

* `INPUT_DIR`: Path to interaction results (e.g., `../result/inter/clinic`).
* `OUTPUT_DIR`: Path for evaluation results (e.g., `../result/eval/clinic`).
* `PYTHON_SCRIPT`: Path to the evaluation script (e.g., `../workflow/evaluation.py`).
* `DATASET`: Dataset name (e.g., `clinic`).
* `MODEL`: Model name (e.g., `google/gemini-3-flash-preview`).
* `KEY`: Your API key ("to be filled").
* `OPENAI_BASE_URL`: Export your base URL ("to be filled").

---

### Suggested Next Step

Would you like me to generate a `requirements.txt` template based on the libraries mentioned in your code snippets (like `pandas`, `matplotlib`, etc.) to complete the repository setup?