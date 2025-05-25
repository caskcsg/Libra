# 🎭 Libra-Emo Dataset

> Libra-Emo is a **large-scale multimodal fine-grained dataset** for **negative emotion detection**. It includes video clips, subtitles, emotion labels, and corresponding explanations.

## 📊 Dataset Description

### 📝 Sample Structure
**Each sample includes:**
- 🎥 A video clip
- 💬 The corresponding subtitle
- 🏷️ An emotion label
- 📋 An explanation of why the label was assigned

### 🎯 Emotion Categories

#### 😊 Positive Emotions
- **Excited** 😆: A high-energy, positive state marked by eagerness, anticipation, and enthusiasm.
- **Happy** 😀: A general sense of contentment, joy, or life satisfaction, often calm and sustained.
- **Amazed** 😲: A strong and lasting sense of wonder or astonishment, often triggered by something extraordinary or impressive.

#### 😐 Neutral Emotions
- **Surprised** 😮: An immediate emotional response to an unexpected event, marked by sudden awareness or shock.
- **Neutral** 😐: An emotionally unmarked state, indicating neither positive nor negative affect.

#### 😔 Negative Emotions
- **Ironic** 🫤: A sarcastic or mocking emotional state, often marked by indirect critique or contradiction.
- **Disgusted** 😒: A visceral reaction of revulsion or strong disapproval, often in response to something morally or physically offensive.
- **Frustrated** 😩: A state of tension, irritation, and dissatisfaction resulting from obstacles that prevent achieving goals or expectations.
- **Angry** 😡: A reactive emotion involving displeasure, irritation, or aggression, usually in response to perceived wrongs or injustice.
- **Hateful** 😠: A persistent, intense hostility or contempt directed toward a person, group, or idea, often associated with a desire for harm.
- **Fearful** 😱: A defensive emotion involving anxiety or dread, triggered by perceived threats or uncertainty.
- **Sad** 😢: A low-energy emotion characterized by feelings of loss, disappointment, or helplessness.
- **Despairful** 😞: A profound sense of hopelessness and helplessness, often accompanied by emotional distress and loss of purpose.

## 📥 Download Instructions

To download the dataset, you need to use Git LFS (Large File Storage):

```bash
# Install Git LFS
git lfs install

# Clone the repository
git clone https://huggingface.co/datasets/caskcsg/Libra-Emo
```

After cloning, the dataset files will be automatically downloaded through Git LFS.

## 📦 Extract Dataset Files

The dataset files are compressed in `tar` format. You can use the provided `extract_tar_files.py` script to extract them:

```bash
# Basic usage (extract to current directory and remove tar files)
python extract_tar_files.py --dir /path/to/tar/files

# Extract to a specific directory
python extract_tar_files.py --dir /path/to/tar/files --target /path/to/extract

# Keep tar files after extraction
python extract_tar_files.py --dir /path/to/tar/files --keep-tar

# Combine options
python extract_tar_files.py --dir /path/to/tar/files --target /path/to/extract --keep-tar
```

### ⚙️ Script Options

| Option | Description |
|--------|-------------|
| `--dir` | Directory containing tar files (default: current directory) |
| `--target` | Target directory to extract files to (optional) |
| `--keep-tar` | Keep tar files after extraction (optional, by default tar files are removed) |

The script will automatically create the target directory if it doesn't exist.

## 🔍 Evaluation

### 📥 Model Download

We provide four different model sizes for various computational requirements. You can download the models from Hugging Face:

| Model | Size | Base Model | Performance |
|-------|------|------------|-------------|
| [Libra-Emo-1B](https://huggingface.co/caskcsg/Libra-Emo-1B) | 1B | InternVL-2.5-1B | 53.54% Acc, 50.19% Weighted-F1 |
| [Libra-Emo-2B](https://huggingface.co/caskcsg/Libra-Emo-2B) | 2B | InternVL-2.5-2B | 56.38% Acc, 53.90% Weighted-F1 |
| [Libra-Emo-4B](https://huggingface.co/caskcsg/Libra-Emo-4B) | 4B | InternVL-2.5-4B | 65.20% Acc, 64.41% Weighted-F1 |
| [Libra-Emo-8B](https://huggingface.co/caskcsg/Libra-Emo-8B) | 8B | InternVL-2.5-8B | 71.18% Acc, 70.71% Weighted-F1 |


### 🚀 Prediction

You can use the provided `predict.py` script to run predictions on the dataset. The script supports both single GPU and multi-GPU inference.

```bash
# Single GPU prediction
python predict.py \
    --model_path caskcsg/Libra-Emo-8B \
    --data_path ./test.json \
    --video_root ./test \
    --output_root ./results/libra-emo-8b \
    --max_tokens 10

# Multi-GPU prediction (e.g., using 2 GPUs)
# Run on GPU 0
CUDA_VISIBLE_DEVICES=0 python predict.py \
    --model_path caskcsg/Libra-Emo-8B \
    --data_path ./test.json \
    --video_root ./test \
    --output_root ./results/libra-emo-8b \
    --machine_num 2 \
    --machine_rank 1 \
    --max_tokens 10

# Run on GPU 1
CUDA_VISIBLE_DEVICES=1 python predict.py \
    --model_path caskcsg/Libra-Emo-8B \
    --data_path ./test.json \
    --video_root ./test \
    --output_root ./results/libra-emo-8b \
    --machine_num 2 \
    --machine_rank 2 \
    --max_tokens 10
```

### ⚙️ Script Options

| Option | Description |
|--------|-------------|
| `--model_path` | Path to the model directory (e.g., "caskcsg/Libra-Emo-8B") |
| `--data_path` | Path to the test data JSON file (e.g., "Libra-Emo/test.json") |
| `--video_root` | Root directory containing the video files (e.g., "Libra-Emo/test") |
| `--output_root` | Directory to save prediction results |
| `--seed` | Random seed for shuffling (default: 42) |
| `--machine_num` | Number of GPUs to use (default: 1) |
| `--machine_rank` | Rank of current GPU (1-based, default: 1) |
| `--max_tokens` | Maximum number of tokens to generate (default: 10) |

The script will:
1. 📥 Load the test dataset
2. 🔄 Shuffle the data (if using multiple GPUs, each GPU gets a shard)
3. 🎯 Process each video and generate predictions
4. 💾 Save results in JSONL format in the output directory

Each output file will be named `{machine_rank}-of-{machine_num}.jsonl` and contain predictions in the format:
```json
{
    "video": "path/to/video",
    "subtitle": "video subtitle",
    "label": "ground truth label",
    "explanation": "ground truth explanation",
    "answer": "model prediction"
}
```

### 📊 Evaluation

After running predictions, you can evaluate the model's performance using the provided `evaluate.py` script:

```bash
python evaluate.py --predict_root ./results/libra-emo-8b
```

The script will compute and display the following metrics:
- 📈 Overall accuracy
- 📊 Overall macro F1 score
- 📉 Overall weighted F1 score
- 😔 Negative emotions accuracy
- 📊 Negative emotions macro F1 score
- 📉 Negative emotions weighted F1 score

The negative emotions category includes: ironic, frustrated, disgusted, sad, angry, fearful, hateful, and despairful.

The script will also print a detailed classification report showing precision, recall, and F1-score for each emotion category.