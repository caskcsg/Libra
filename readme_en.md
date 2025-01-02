# Libra Guard: Large Chinese-based Safeguard for AI Content

**Libra Guard** is a safeguard model designed for large Chinese language models (LLMs). Libra Guard employs a two-stage progressive training process, first utilizing scalable synthetic samples for pre-training, and then fine-tuning with high-quality real data. This approach maximizes data utilization while reducing reliance on manual annotation. Additionally, Libra Guard establishes the first safety benchmark specifically for Chinese LLMs — **Libra Bench**. Experiments demonstrate that Libra Guard significantly outperforms similar open-source models (such as ShieldLM) on Libra Bench and is competitive with advanced commercial models (like GPT-4o) across multiple tasks, providing stronger support and evaluation tools for the safety governance of Chinese LLMs.

---

## ✨ Key Features

1. **Two-Stage Curriculum Training**: Initially pre-trains with scalable synthetic samples, then fine-tunes with real data, enhancing model performance while reducing dependence on manual annotations.

2. **Scalable Data Pipeline**: Combines real and synthetic data to minimize manual annotation workload and extensively cover potential safety risks.

3. **High Performance**: Significantly outperforms other open-source models in accuracy and remains competitive against closed-source models.

4. **Libra Bench**: The first safety evaluation benchmark for Chinese large models, covering seven key risk scenarios and over 5,700 expert-annotated data points.

---

## 📊 Libra Bench

**Libra Bench** is a safety evaluation benchmark specifically constructed for Chinese large models, encompassing the following three data sources and undergoing rigorous manual review:

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

You can download and load the **Libra Bench** dataset from the Hugging Face Hub. The format is as follows:

```json
{
  "source": "query source",
  "category_name": "harmful category of query",
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
| `--is_shieldlm`    | bool | False   | Whether to parse using the `ShieldLM` format; if true, use `get_predict_shieldLM()` function |

### Usage Example

Please refer to the following command to run:

```bash
python ./scripts/evaluate_metrics.py \
    --predict_root ./outputs \
    --label_path ./data/test.json \
    --is_shieldlm False
```

- `--predict_root`: Specifies the inference output folder. The script will automatically read all result files in this folder (e.g., `0-of-1.jsonl`).
- `--label_path`: Path to the test set label file (JSON format, containing fields like `id` and `label` for each data point).
- `--is_shieldlm`: Defaults to `False`. If your inference output format differs from ShieldLM, set this to `True`.

After the script finishes running, you will see output similar to the following metrics:

```
Total: 5720
Errors: 300
Average Accuracy: 0.9476
synthesis :  0.9453
Safety-Prompts :  0.9365
BeaverTails_30k :  0.9591
```

- `Total`: Total number of samples aligned with inference results and labels
- `Errors`: Number of samples where parsing failed or labels do not match predictions
- `Average Accuracy`: Overall accuracy
- Accuracy per scenario: e.g., `synthesis`, `Safety-Prompts`, `BeaverTails_30k`, etc.

**With the inference script and metric calculation script, you can complete the end-to-end evaluation process: from generating inference results to final safety metric statistical analysis.**

---

## 📊 Experimental Results

In the following table, we evaluated various baseline models (Instruct models and Guard models) and compared their performance with **Libra Guard**.

| **Models**                       | **Average** | **Synthesis** | **Safety-Prompts** | **BeaverTails\_30k** |
|----------------------------------|-------------|---------------|--------------------|----------------------|
| **Instruct Models**              |             |               |                    |                      |
| Qwen-14B-Chat                    | 0.6883      | 0.5787        | 0.6886             | 0.7977               |
| Qwen2.5-0.5B-Instruct            | 0.6337      | 0.5740        | 0.6482             | 0.6790               |
| Qwen2.5-1.5B-Instruct            | 0.6530      | 0.5719        | 0.6648             | 0.7222               |
| Qwen2.5-3B-Instruct              | 0.7121      | 0.6360        | 0.7060             | 0.7944               |
| Qwen2.5-7B-Instruct              | 0.6249      | 0.5392        | 0.5563             | 0.7793               |
| Qwen2.5-14B-Instruct             | 0.7433      | 0.6810        | 0.6696             | 0.8793               |
| Yi-1.5-9B-Chat                   | 0.5174      | 0.4097        | 0.4334             | 0.7091               |
| **Guard Models**                 |             |               |                    |                      |
| Llama-Guard                      | 0.3961      | 0.3388        | 0.2845             | 0.5650               |
| ShieldGemma                      | 0.4403      | 0.4104        | 0.3154             | 0.5951               |
| ShieldLM                         | 0.6569      | 0.6196        | 0.5341             | 0.8171               |
| Libra-Guard-Qwen-14B-Chat        | 0.8648      | 0.8296        | 0.8534             | 0.9114               |
| Libra-Guard-Qwen2.5-0.5B-Instruct | 0.8146      | 0.7905        | 0.8223             | 0.8311               |
| Libra-Guard-Qwen2.5-1.5B-Instruct | 0.8392      | 0.7975        | 0.8376             | 0.8826               |
| Libra-Guard-Qwen2.5-3B-Instruct   | 0.8475      | 0.8153        | 0.8391             | 0.8880               |
| Libra-Guard-Qwen2.5-7B-Instruct   | 0.8524      | 0.8132        | 0.8471             | 0.8970               |
| Libra-Guard-Qwen2.5-14B-Instruct  | **0.8679**  | 0.8337        | 0.8597             | 0.9104               |
| Libra-Guard-Yi-1.5-9B-Chat        | 0.8593      | 0.8200        | 0.8645             | 0.8933               |

Libra Guard significantly outperforms Instruct and Guard baselines in safety detection tasks, demonstrating its strong performance across multiple benchmarks and data types.

---

## 📝 Citation

If this project is helpful to you, please cite the following papers:

```bibtex
@article{chen2024libra,
  title={Libra Guard: Large Chinese-based Safeguard for AI Content},
  author={Chen, Ziyang and Yu, Huimu and Wu, Xing and Hu, Songlin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
@inproceedings{chen2024libra,
    title = "{L}ibra {G}uard: A Large Chinese-based Safeguard for AI Content",
    author = "Chen, Ziyang and Yu, Huimu and Wu, Xing and Hu, Songlin",
    booktitle = "Proceedings of the XXXth Conference on Computational Linguistics",
    month = aug,
    year = "2024",
    address = "City, Country",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2024.conll-XXX.XXX",
    pages = "XXXX--XXXX",
}
```
