## Multi-View Summarization using T5 with LoRA and Custom Input Fusion

This repository contains a Jupyter Notebook (`best4_model_ipynb.ipynb`) for fine-tuning a `T5-Large` model on **mental health counselling dialogue data** using a novel **multi-view summarization** approach.   
The summarization incorporates **speaker** and **emotion** views, and leverages **LoRA (Low-Rank Adaptation)** via the PEFT library to improve performance efficiently.

---

## 📁 Project Structure

```
├── best4_model_ipynb.ipynb          # Main notebook for training and inference
├── train_preprocess_v2.json         # Preprocessed training dataset
├── val_preprocess_v2.json           # Preprocessed validation dataset
├── test_preprocess_v2.json          # Preprocessed test dataset
├── model_checkpoints/               # Saved model checkpoints (generated during training)
├── logs/                            # Training and evaluation logs
```

---

## ⚙️ Setup Instructions

Install all necessary dependencies:

```bash
pip install rouge_score
pip install bert_score
pip install evaluate
pip install sacrebleu
pip install torch transformers peft accelerate datasets
```

You'll also need:

```python
import json
import random
import sacrebleu
import nltk
import torch
import os
import evaluate
import sys
from tqdm.notebook import tqdm

from datasets import Dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
from transformers.utils import logging as hf_logging
from bert_score import score as bert_score
from rouge_score import rouge_scorer
```

Ensure you have access to the training and validation JSON files used in the notebook.

---

## ▶️ How to Run the Notebook

1. Launch Jupyter Notebook or use an online environment like **Google Colab** or **Kaggle**.

2. Upload or mount your dataset files to the correct file paths expected by the notebook:

```python
train_data = load_data('/kaggle/input/preprocessed-new-aniket/train_preprocess_v2.json')
val_data   = load_data('/kaggle/input/preprocessed-new-aniket/val_preprocess_v2.json')
```

 You can change the paths based on your local or cloud environment.

3. Run all cells sequentially to:
- Load and preprocess the dataset  
- Perform multi-view input fusion (speaker and emotion)  
- Fine-tune the T5 model using LoRA  
- Generate **overview** and **detailed** summaries  
- Evaluate using **BLEU**, **ROUGE**, and **BERTScore**

---

## ✨ Model Highlights

- **Multi-View Input Fusion**: Integrates speaker and emotion context into the input sequence
- **T5-Large**: Pretrained transformer model used for summarization
- **LoRA + PEFT**: Efficient fine-tuning via Low-Rank Adaptation (LoRA) using Hugging Face's PEFT library
- **Custom Evaluation**: Uses `evaluate`, `sacrebleu`, `roguescore` and `bert_score` for evaluation.
- **Manual Inspection**: Epoch-wise train vs val loss and generation of outputs.

---

## 📊 Output & Evaluation

- The notebook **saves the fine-tuned model**.
- The notebook **saves the fine-tuned model checkpoint**
- Sample predictions are displayed alongside ground truth summaries
- **Train and validation loss curves** are plotted to assess model performance
- Final outputs are evaluated using:
  - ✅ **BLEU**
  - ✅ **ROUGE**
  - ✅ **BERTScore**

---

## 📝 Notes

- The current version assumes specific file paths. Please modify paths as needed based on your environment.
- Model outputs include both **overview** and **detailed** summaries.
- Manual epoch-wise evaluation was conducted to monitor training effectiveness.

---

## 📎 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [ROUGE Score](https://pypi.org/project/rouge-score/)
- [BERTScore](https://github.com/Tiiiger/bert_score)
