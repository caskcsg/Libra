# Libra：面向中文大模型的安全护栏模型 (Libra: Large Chinese-based Safeguard for AI Content)

**Libra-Guard** 是一款面向中文大型语言模型（LLM）的安全护栏模型。Libra-Guard 采用两阶段渐进式训练流程，先利用可扩展的合成样本预训练，再使用高质量真实数据进行微调，最大化利用数据并降低对人工标注的依赖，并构建了首个针对中文 LLM 的安全基准 —— **Libra-Test**。实验表明，Libra-Guard 在 Libra-Test 上的表现显著优于同类开源模型（如 ShieldLM等），在多个任务上可与先进商用模型（如 GPT-4o）接近，为中文 LLM 的安全治理提供了更强的支持与评测工具。

**模型:**  [Libra-Guard](https://huggingface.co/collections/caskcsg/libra-guard-67765779999dab7ca25180a2)

**数据集:**  [Libra-Test](https://huggingface.co/datasets/caskcsg/Libra-Test)

![Libra-Guard](https://github.com/caskcsg/Libra/blob/main/LibraGuard.png)

---



## ✨ 主要特性 (Key Features)

1. **两阶段课程式训练**：先利用可扩展的合成样本进行预训练，再用真实数据微调，在减少人工标注依赖的同时提升模型表现。  

2. **可扩展数据流程**：结合真实与合成数据，减少人工标注工作量并广泛覆盖潜在安全风险。  

3. **高性能表现**：在准确率上显著优于其它开源模型，对比闭源模型亦具有竞争力。  

4. **Libra-Test**：首个中文大模型安全评测基准，涵盖七大关键风险场景和 5,700+ 条专家标注数据。  

---

## 📊 Libra-Test

**Libra-Test** 是专为中文大模型安全性而构建的评测基准，涵盖以下三种数据来源并经过严格的人工审核：  

1. **真实数据（Real Data）**  
2. **合成数据（Synthetic Data）**  
3. **翻译数据（Translated Data）**



| **Type**          | **Safe** | **Unsafe** | **Total** |
|-------------------|----------|------------|-----------|
| Real Data         | 381      | 881        | 1,262     |
| Synthetic Data    | 583      | 884        | 1,467     |
| Translated Data   | 900      | 2,091      | 2,991     |
| **Total**         | **1,864**| **3,856**  | **5,720** |


![Libra-Test](https://github.com/caskcsg/Libra/blob/main/Libra-Test.png)

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
| `--is_shieldlm`                  | bool       | False            | 是否采用 `ShieldLM` 格式解析，若为真则使用 `get_predict_shieldLM()` 函数 |


### 使用示例 (Usage Example)

请参考如下命令运行：

```bash
python ./scripts/evaluate_metrics.py \
    --predict_root ./outputs \
    --label_path ./data/test.json \
    --is_shieldlm False
```

- `--predict_root` 指定推理输出文件夹。脚本会自动读取该文件夹内所有结果文件（如 `0-of-1.jsonl`等）。  
- `--label_path` 为测试集的标签文件（JSON 格式，包含每条数据的 `id` 及 `label` 等字段）。  
- `--is_shieldlm` 默认为 `False`，若您的推理输出与 ShieldLM 的格式不同，可设为 `True`。  

脚本运行结束后，您将看到类似如下的指标输出：

```
总数： 5720
错误数： 300
平均准确率： 0.9476
synthesis :  0.9453
Safety-Prompts :  0.9365
BeaverTails_30k :  0.9591
```

- `总数`：推理结果与标注对齐后的总样本数  
- `错误数`：无法正确解析或标签与预测不一致的数据数目  
- `平均准确率`：整体正确率  
- 各场景准确率：如 `synthesis`, `Safety-Prompts`, `BeaverTails_30k` 等  



**至此，配合推理脚本和评测指标脚本，您就可以完成端到端的评测流程：从推理结果的生成到最终安全指标的统计分析。**  


---

## 📊 实验结果 (Experimental Results)

在以下表格中，我们对多种基线模型（Instruct 模型与 Guard 模型）进行了评测，并与 **Libra-Guard** 的表现进行了对比。  

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

Libra-Guard 在安全检测任务中显著优于 Instruct 和 Guard 基线，展示了其在多个基准和数据集类型上的强大性能。


---



## 📝 引用 (Citation)

如果本项目对您有帮助，请引用以下论文：  

```bibtex
@article{chen2024libra,
  title={Libra: Large Chinese-based Safeguard for AI Content},
  author={Chen, Ziyang and Yu, Huimu and Wu, Xing and Hu, Songlin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
@inproceedings{chen2024libra,
    title = "{L}ibra: A Large Chinese-based Safeguard for AI Content",
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
