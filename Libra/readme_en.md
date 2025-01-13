# Libra: Large Chinese-based Safeguard for AI Content

**Libra-Guard** is a safeguard model designed for large Chinese language models (LLMs). Libra-Guard employs a two-stage progressive training process, first utilizing scalable synthetic samples for pre-training, and then fine-tuning with high-quality real data. This approach maximizes data utilization while reducing reliance on manual annotation. Additionally, Libra-Guard establishes the first safety benchmark specifically for Chinese LLMs — **Libra-Test**. Experiments demonstrate that Libra-Guard significantly outperforms similar open-source models (such as ShieldLM) on Libra-Test and is competitive with advanced commercial models (like GPT-4o) across multiple tasks, providing stronger support and evaluation tools for the safety governance of Chinese LLMs.

**Model:**  [Libra-Guard](https://huggingface.co/collections/caskcsg/libra-guard-67765779999dab7ca25180a2)

**Test Set:**  [Libra-Test](https://huggingface.co/datasets/caskcsg/Libra-Test)

![Libra-Guard](https://github.com/caskcsg/Libra/blob/main/LibraGuard.png)

---

## ✨ Key Features

1. **Two-Stage Curriculum Training**: Initially pre-trains with scalable synthetic samples, then fine-tunes with real data, enhancing model performance while reducing dependence on manual annotations.

2. **Scalable Data Pipeline**: Combines real and synthetic data to minimize manual annotation workload and extensively cover potential safety risks.

3. **High Performance**: Significantly outperforms other open-source models in accuracy and remains competitive against closed-source models.

4. **Libra-Test**: The first safety evaluation benchmark for the safeguards of Chinese LLMs, covering seven key risk scenarios and over 5,700 expert-annotated data points.

---

## 📊 Libra-Test

**Libra-Test** is a safety evaluation benchmark specifically constructed for the safeguards of Chinese LLMs, encompassing the following three data sources and undergoing rigorous manual review:

1. **Real Data**
2. **Synthetic Data**
3. **Translated Data**

| **Type**          | **Safe** | **Unsafe** | **Total** |
|-------------------|----------|------------|-----------|
| Real Data         | 381      | 881        | 1,262     |
| Synthetic Data    | 583      | 884        | 1,467     |
| Translated Data   | 900      | 2,091      | 2,991     |
| **Total**         | **1,864**| **3,856**  | **5,720** |

---

## 🏁 Getting Started

Below are the brief steps to use this project and its provided scripts for inference and evaluation.

### 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/caskcsg/Libra.git
   cd Libra
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Data Loading & Format

You can download and load the **Libra-Test** dataset from the Hugging Face Hub. The format is as follows:

```json
{
  "source": "query source",
  "category_name": "harmful category of query", 
  "sub_category_name": "harmful subcategory of query",
  "question": "query",
  "response": "response",
  "label": "safety label",
  "analyze": "safety analysis",
  "id": "id"
}
```

---

## 🤖 Inference & Evaluation

### Result Inference

This project provides the following script examples to help you perform inference in your local environment:

The script is named `inference.py`, with the main parameters as follows:

| Parameter         | Type | Default | Description                                                                   |
|-------------------|------|---------|-------------------------------------------------------------------------------|
| `--base_model`    | str  | (Required) | Specify the model path or name, e.g., `./models/Libra-Guard-7B`               |
| `--data_path`     | str  | (Required) | Specify the data file for inference or evaluation (JSON / Parquet / folder)  |
| `--batch_size`    | int  | 1       | Inference batch size, adjust appropriately based on GPU memory and data size |
| `--seed`          | int  | 42      | Random seed to ensure reproducibility                                         |
| `--out_path`      | str  | `./outputs` | Path to save inference results                                               |
| `--num_itera`     | int  | 1       | Set to >1 for multiple generations or comparison tests, usually default 1    |
| `--few_shot`      | int  | 0       | Number of few-shot examples; if >0, add demonstration examples before input   |
| `--is_instruct`   | int  | 1       | Whether to use instruction-style inference (enable Instruct template)         |
| `--machine_rank`  | int  | 0       | Current node index (used in distributed settings)                            |
| `--machine_num`   | int  | 1       | Total number of nodes (used in distributed settings)                         |

**Usage example:**

```bash
# Single machine, single GPU scenario
python ./scripts/inference.py \
  --base_model ./models/Libra-Guard-7B \
  --data_path ./data/test.json \
  --batch_size 2 \
  --out_path ./outputs \
  --seed 42
```

Below is an example supplement for the "Result Metrics Calculation" or "Result Statistics" section in a sample **README**, which can be merged with the previous inference script instructions and modified according to your project needs.

---

### Result Metrics Calculation

After completing inference, you can use the **metric calculation script** provided by this project (e.g., `evaluate_metrics.py`) to calculate accuracy and other metrics for safety judgments. This script will compute and display overall and per-scenario accuracy based on your inference result folder and the annotated test set labels.

Below is a brief introduction to the main parameters and functions in the script:

| Parameter          | Type | Default | Description                                                        |
|--------------------|------|---------|--------------------------------------------------------------------|
| `--predict_root`   | str  | (Required) | Path to the folder containing inference results; the script reads all `.jsonl` or `.json` files in this folder |
| `--label_path`     | str  | (Required) | Path to the test set label file (JSON)                             |

### Usage Example

Please refer to the following command to run:

```bash
python ./scripts/evaluate_metrics.py \
    --predict_root ./outputs \
    --label_path ./data/test.json
```

- `--predict_root`: Specifies the inference output folder. The script will automatically read all result files in this folder (e.g., `0-of-1.jsonl`).
- `--label_path`: Path to the test set label file (JSON format, containing fields like `id` and `label` for each data point).

After the script finishes running, you will see output similar to the following metrics (including the Accuracy and F1 scores of three different sources and their average):

```
Real Data
F1-Unsafe: 0.8983
F1-Safe: 0.7739
Accuracy: 0.8597
Synthetic Data
F1-Unsafe: 0.8606
F1-Safe: 0.7939
Accuracy: 0.8337
Translated Data
F1-Unsafe: 0.9359
F1-Safe: 0.8513
Accuracy: 0.9104
Average
F1-Unsafe: 0.8983
F1-Safe: 0.8064
Accuracy: 0.8679
```

**With the inference script and metric calculation script, you can complete the end-to-end evaluation process: from generating inference results to final safety metric statistical analysis.**

---

## 📊 Experimental Results

In the following table, we evaluated various baseline models (Instruct models and Guard models) and compared their performance with **Libra-Guard**.

| **Models**                              |                           |  **Average**  |                     | **Real Data**        | **Synthetic Data**   | **Translated Data**  |
|-----------------------------------------|---------------------------|---------------|---------------------|----------------------|----------------------|----------------------|
|                                         | **Accuracy**              | **$F_1$-Safe**| **$F_1$-Unsafe**    | **Accuracy**         | **Accuracy**         | **Accuracy**         |
| **Closed-Source Models**                |                           |               |                     |                      |                      |                      |
| GPT-4o                                  | 91.05%                    | 87.1%         | 93.04%              | 88.59%               | 89.78%               | 94.78%               |
| Sonnet-3.5                              | 88.82%                    | 82.34%        | 91.77%              | 88.83%               | 84.46%               | 93.18%               |
| **Instruct Models**                     |                           |               |                     |                      |                      |                      |
| Qwen-14B-Chat                           | 68.83%                    | 30.55%        | 79.79%              | 68.86%               | 57.87%               | 79.77%               |
| Qwen2.5-0.5B-Instruct                   | 63.37%                    | 6.47%         | 77.14%              | 64.82%               | 57.4%                | 67.9%                |
| Qwen2.5-1.5B-Instruct                   | 65.3%                     | 34.48%        | 75.84%              | 66.48%               | 57.19%               | 72.22%               |
| Qwen2.5-3B-Instruct                     | 71.21%                    | 49.06%        | 79.74%              | 70.6%                | 63.6%                | 79.44%               |
| Qwen2.5-7B-Instruct                     | 62.49%                    | 59.96%        | 64.09%              | 55.63%               | 53.92%               | 77.93%               |
| Qwen2.5-14B-Instruct                    | 74.33%                    | 65.99%        | 79.32%              | 66.96%               | 68.1%                | 87.93%               |
| Yi-1.5-9B-Chat                          | 51.74%                    | 54.07%        | 47.31%              | 43.34%               | 40.97%               | 70.91%               |
| **Guard Models**                        |                           |               |                     |                      |                      |                      |
| Llama-Guard3-8B                         | 39.61%                    | 48.09%        | 26.1%               | 28.45%               | 33.88%               | 56.5%                |
| ShieldGemma-9B                          | 44.03%                    | 54.51%        | 23.02%              | 31.54%               | 41.04%               | 59.51%               |
| ShieldLM-Qwen-14B-Chat                  | 65.69%                    | 65.24%        | 65.23%              | 53.41%               | 61.96%               | 81.71%               |
| Libra-Guard-Qwen-14B-Chat               | 86.48%                    | 80.58%        | 89.51%              | 85.34%               | 82.96%               | 91.14%               |
| Libra-Guard-Qwen2.5-0.5B-Instruct       | 81.46%                    | 69.29%        | 86.26%              | 82.23%               | 79.05%               | 83.11%               |
| Libra-Guard-Qwen2.5-1.5B-Instruct       | 83.93%                    | 77.13%        | 87.37%              | 83.76%               | 79.75%               | 88.26%               |
| Libra-Guard-Qwen2.5-3B-Instruct         | 84.75%                    | 78.01%        | 88.13%              | 83.91%               | 81.53%               | 88.8%                |
| Libra-Guard-Qwen2.5-7B-Instruct         | 85.24%                    | 79.41%        | 88.33%              | 84.71%               | 81.32%               | 89.7%                |
| Libra-Guard-Qwen2.5-14B-Instruct        | **86.79%**                | **80.64%**    | **89.83%**          | 85.97%               | 83.37%               | 91.04%               |
| Libra-Guard-Yi-1.5-9B-Chat              | 85.93%                    | 79.15%        | 89.2%               | 86.45%               | 82%                  | 89.33%               |

Libra-Guard significantly outperforms Instruct and Guard baselines in safety detection tasks, demonstrating its strong performance across multiple benchmarks and data types.

---

## 📝 Citation

If this project is helpful to you, please cite the following papers:

```bibtex
@misc{libra,
    title = {Libra: Large Chinese-based Safeguard for AI Content},
    url = {https://github.com/caskcsg/Libra/},
    author= {Chen, Ziyang and Yu, Huimu and Wu, Xing and Lin, Yuxuan and Liu, Dongqin and Hu, Songlin},
    month = {January},
    year = {2025}
}
```
