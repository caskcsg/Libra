# Libra：面向中文大模型的安全护栏模型 (Libra: Large Chinese-based Safeguard for AI Content)

**Libra-Guard** 是一款面向中文大型语言模型（LLM）的安全护栏模型。Libra-Guard 采用两阶段渐进式训练流程，先利用可扩展的合成样本预训练，再使用高质量真实数据进行微调，最大化利用数据并降低对人工标注的依赖，并构建了首个针对中文 LLM 的安全基准 —— **Libra-Test**。实验表明，Libra-Guard 在 Libra-Test 上的表现显著优于同类开源模型（如 ShieldLM等），在多个任务上可与先进商用模型（如 GPT-4o）接近，为中文 LLM 的安全治理提供了更强的支持与评测工具。

**模型:**  [Libra-Guard](https://huggingface.co/collections/caskcsg/libra-guard-67765779999dab7ca25180a2)

**评测集:**  [Libra-Test](https://huggingface.co/datasets/caskcsg/Libra-Test)

![Libra-Guard](https://github.com/caskcsg/Libra/blob/main/Libra/LibraGuard.png)

---



## ✨ 主要特性 (Key Features)

1. **两阶段课程式训练**：先利用可扩展的合成样本进行预训练，再用真实数据微调，在减少人工标注依赖的同时提升模型表现。  

2. **可扩展数据流程**：结合真实与合成数据，减少人工标注工作量并广泛覆盖潜在安全风险。  

3. **高性能表现**：在准确率上显著优于其它开源模型，对比闭源模型亦具有竞争力。  

4. **Libra-Test**：首个中文大模型护栏评测基准，涵盖七大关键风险场景和 5,700+ 条专家标注数据。  

---

## 📊 Libra-Test

**Libra-Test** 是专为中文大模型护栏而构建的评测基准，涵盖以下三种数据来源并经过严格的人工审核：  

1. **真实数据（Real Data）**  
2. **合成数据（Synthetic Data）**  
3. **翻译数据（Translated Data）**



| **Type**          | **Safe** | **Unsafe** | **Total** |
|-------------------|----------|------------|-----------|
| Real Data         | 381      | 881        | 1,262     |
| Synthetic Data    | 583      | 884        | 1,467     |
| Translated Data   | 900      | 2,091      | 2,991     |
| **Total**         | **1,864**| **3,856**  | **5,720** |


![Libra-Test](https://github.com/caskcsg/Libra/blob/main/Libra/Libra-Test.png)

---

## 🏁 快速开始 (Getting Started)

下面列出了使用本项目及其提供的脚本进行推理与评测的简要步骤。

### 🛠️ 环境安装 (Installation)

1. 克隆本仓库：  
   ```bash
   git clone https://github.com/caskcsg/Libra.git
   cd Libra
   ```
2. 安装依赖：  
   ```bash
   pip install -r requirements.txt
   ```
---

## 📥 数据加载与格式 (Data Loading & Format)

您可以从 Hugging Face Hub下载并加载 **Libra-Test** 数据集，格式如下所示。  

```json
{
  "source": "query来源",
  "category_name": "query有害类别",
  "sub_category_name": "query有害子类别",
  "question": "query",
  "response": "response",
  "label": "安全标签",
  "analyze": "安全分析",
  "id": "id"
}
```


---

## 🤖 推理与评测 (Inference & Evaluation)

### 结果推理 (Result Inference)

本项目提供了如下脚本示例，帮助您在本地环境中完成推理：



脚本名为 `inference.py`，其主要参数如下：  

| 参数 (Parameter)     | 类型 (Type) | 默认值 (Default) | 说明 (Description)                                                                   |
|----------------------|-------------|------------------|--------------------------------------------------------------------------------------|
| `--base_model`       | str         | (必填 Required)  | 指定模型路径或名称，如 `./models/Libra-Guard-7B`                                    |
| `--data_path`        | str         | (必填 Required)  | 指定待推理或评测的数据文件（JSON / Parquet / 文件夹）                                |
| `--batch_size`       | int         | 1                | 推理批大小，视 GPU 显存和数据量适当调整                                               |
| `--seed`             | int         | 42               | 随机种子，保证结果可复现                                                              |
| `--out_path`         | str         | `./outputs`      | 推理结果保存路径                                                                     |
| `--num_itera`        | int         | 1                | 多次生成或对比测试时可设置为 >1，一般默认 1                                          |
| `--few_shot`         | int         | 0                | few-shot 示例数，如 >0 则在输入前添加演示示例                                         |
| `--is_instruct`      | int         | 1                | 是否采用指令风格推理（启用 Instruct 模板）                                           |
| `--machine_rank`     | int         | 0                | 当前节点序号（分布式时使用）                                                          |
| `--machine_num`      | int         | 1                | 节点总数（分布式时使用）                                                              |



**使用示例：**  
*Usage example:*

```bash
# 单机单卡场景
python ./scripts/inference.py \
  --base_model ./models/Libra-Guard-7B \
  --data_path ./data/test.json \
  --batch_size 2 \
  --out_path ./outputs \
  --seed 42
```

下面是一段示例 **README** 中“评测指标计算”或“结果统计”部分的补充示例，可与之前的推理脚本说明合并使用，并根据您的项目情况进行相应的修改。

---

###  评测指标计算 (Result Metrics Calculation)

在完成推理后，您可以使用本项目提供的 **计算指标脚本**（如 `evaluate_metrics.py`）来统计安全判断的准确率等指标。该脚本会根据您的推理结果文件夹以及标注好的测试集标签，计算并打印出整体与分场景的准确率等信息。


下面简要介绍脚本中的主要参数和功能：

| 参数 (Parameter)                 | 类型 (Type) | 默认值 (Default) | 说明 (Description)                                                      |
|----------------------------------|------------|------------------|-------------------------------------------------------------------------|
| `--predict_root`                 | str        | (必填) Required  | 推理结果所在文件夹路径，脚本会读取该文件夹中所有 `.jsonl` 或 `.json` 文件 |
| `--label_path`                   | str        | (必填) Required  | 测试集标签文件（JSON）的路径                                            |

### 使用示例 (Usage Example)

请参考如下命令运行：

```bash
python ./scripts/evaluate_metrics.py \
    --predict_root ./outputs \
    --label_path ./data/test.json
```

- `--predict_root` 指定推理输出文件夹。脚本会自动读取该文件夹内所有结果文件（如 `0-of-1.jsonl`等）。  
- `--label_path` 为测试集的标签文件（JSON 格式，包含每条数据的 `id` 及 `label` 等字段）。  

脚本运行结束后，您将看到类似如下的指标输出（包含三个不同来源以及三者平均的准确率与F1值）：

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


**至此，配合推理脚本和评测指标脚本，您就可以完成端到端的评测流程：从推理结果的生成到最终安全指标的统计分析。**  


---

## 📊 实验结果 (Experimental Results)

在以下表格中，我们对多种基线模型（Closed-Source 模型，Instruct 模型与 Guard 模型）进行了评测，并与 **Libra Guard** 的表现进行了对比。  

| **Models**                              |                           |  **Average**  |                     | **Real**        | **Synthetic**   | **Translated**  |
|-----------------------------------------|---------------------------|---------------|---------------------|----------------------|----------------------|----------------------|
|                                         | **Accuracy**              | **$F_1$-Safe**| **$F_1$-Unsafe**    | **Accuracy**         | **Accuracy**         | **Accuracy**         |
| **Closed-Source Models**                |                           |               |                     |                      |                      |                      |
| GPT-4o                                  | 91.05%                    | 87.1%         | 93.04%              | 88.59%               | 89.78%               | 94.78%               |
| Claude-3.5-Sonnet                       | 88.82%                    | 82.34%        | 91.77%              | 88.83%               | 84.46%               | 93.18%               |
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

Libra Guard 在安全检测任务中显著优于 Instruct 和 Guard 基线，逼近 Closed-Source 基线，展示了其在多个基准和数据集类型上的强大性能。

---



## 📝 引用 (Citation)

如果本项目对您有帮助，请引用以下论文：  

```bibtex
@misc{libra,
    title = {Libra: Large Chinese-based Safeguard for AI Content},
    url = {https://github.com/caskcsg/Libra/},
    author= {Chen, Ziyang and Yu, Huimu and Wu, Xing and Lin, Yuxuan and Liu, Dongqin and Hu, Songlin},
    month = {January},
    year = {2025}
}
```
