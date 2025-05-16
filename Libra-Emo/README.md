# Libra-Emo Dataset

Libra-Emo is a comprehensive video emotion recognition dataset featuring clips from TV shows with corresponding subtitles and emotion labels.

## Dataset Description

The dataset contains video clips labeled with 13 emotion categories:
- excited
- happy
- amazed
- surprised
- neutral
- ironic
- disgusted
- frustrated
- angry
- hateful
- fearful
- sad
- despairful

Each sample includes:
- A video clip
- The corresponding subtitle
- An emotion label
- An explanation of why the label was assigned

## Download Instructions

To download the dataset, you need to use Git LFS (Large File Storage):

```bash
# Install Git LFS
git lfs install

# Clone the repository
git clone https://huggingface.co/datasets/caskcsg/Libra-Emo
```

After cloning, the dataset files will be automatically downloaded through Git LFS.

## Extract Dataset Files

The dataset files are compressed in tar format. You can use the provided `extract_tar_files.py` script to extract them:

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

### Script Options

- `--dir`: Directory containing tar files (default: current directory)
- `--target`: Target directory to extract files to (optional)
- `--keep-tar`: Keep tar files after extraction (optional, by default tar files are removed)

The script will automatically create the target directory if it doesn't exist.

## Evaluation

### Prediction

You can use the provided `predict.py` script to run predictions on the dataset. The script supports both single GPU and multi-GPU inference.

```bash
# Single GPU prediction
python predict.py \
    --model_path /path/to/model \
    --data_path /path/to/test.json \
    --video_root /path/to/videos \
    --output_root /path/to/output \
    --max_tokens 10

# Multi-GPU prediction (e.g., using 4 GPUs)
# Run on GPU 0
python predict.py \
    --model_path /path/to/model \
    --data_path /path/to/test.json \
    --video_root /path/to/videos \
    --output_root /path/to/output \
    --machine_num 2 \
    --machine_rank 1 \
    --max_tokens 10

# Run on GPU 1
python predict.py \
    --model_path /path/to/model \
    --data_path /path/to/test.json \
    --video_root /path/to/videos \
    --output_root /path/to/output \
    --machine_num 2 \
    --machine_rank 2 \
    --max_tokens 10
```

### Script Options

- `--model_path`: Path to the model directory
- `--data_path`: Path to the test data JSON file (e.g., "Libra-Emo/test.json")
- `--video_root`: Root directory containing the video files (e.g., "Libra-Emo/test")
- `--output_root`: Directory to save prediction results
- `--seed`: Random seed for shuffling (default: 42)
- `--machine_num`: Number of GPUs to use (default: 1)
- `--machine_rank`: Rank of current GPU (1-based, default: 1)
- `--max_tokens`: Maximum number of tokens to generate (default: 10)

The script will:
1. Load the test dataset
2. Shuffle the data (if using multiple GPUs, each GPU gets a shard)
3. Process each video and generate predictions
4. Save results in JSONL format in the output directory

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

### Evaluation

After running predictions, you can evaluate the model's performance using the provided `evaluate.py` script:

```bash
python evaluate.py --predict_root /path/to/predictions
```

The script will compute and display the following metrics:
- Overall accuracy
- Overall macro F1 score
- Overall weighted F1 score
- Negative emotions accuracy
- Negative emotions macro F1 score
- Negative emotions weighted F1 score

The negative emotions category includes: ironic, frustrated, disgusted, sad, angry, fearful, hateful, and despairful.

The script will also print a detailed classification report showing precision, recall, and F1-score for each emotion category.












