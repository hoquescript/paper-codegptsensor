# CodeGPTSensor
This repository is for the paper titled "*Distinguishing LLM-generated from Human-written Code by Contrastive Learning*".

<div align="center">
<img src="assets/framework.png" alt="The framework of CodeGPTSensor" width="60%">
<p> Fig. 1. The framework of CodeGPTSensor </p> 
</div>

### Project Structure
``` bash
├── CodeGPTSensor       # Implementation of CodeGPTSensor
|	├── models_output 	# Models will be saved here
|	├── utils           # Code for early stopping
|	├── model.py        # Code for our model
|	└── run.py          # Code for training and evaluation
├── dataset             # Extracted Java and Python datasets
└── dataset.zip         # The Java and Python train/valid/test datasets of ".jsonl" files
```

### Setup
Create a virtual environment in the project root and install dependencies:
``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd CodeGPTSensor
```

### Implementation of CodeGPTSensor
To train CodeGPTSensor, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --do_train \
    --model_name_or_path microsoft/unixcoder-base-nine \
    --train_data_file ../dataset/java/train.jsonl \
    --eval_data_file ../dataset/java/valid.jsonl \
    --output_dir models_output/java \
    --num_train_epochs 20 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 99 \
    --contrast
```

To test CodeGPTSensor, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --do_test \
    --model_name_or_path microsoft/unixcoder-base-nine \
    --output_dir models_output/java \
    --test_data_file ../dataset/java/test.jsonl \
    --block_size 400 \
    --eval_batch_size 16 \
    --seed 99
```

For a quick local smoke test, replace the dataset with `../dataset/python/train_small.jsonl`, set `--num_train_epochs 1`, and use smaller batch sizes such as `--train_batch_size 2 --eval_batch_size 2`.

**Notes:**
- Replace "train.jsonl" with "train_no_comment.jsonl" for the without comment datasets.
- Replace "java" with "python" for the Python datasets.# paper-codegptsensor
