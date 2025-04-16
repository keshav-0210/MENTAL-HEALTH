## Zero-shot Prompting in T5 and MentalLLama using Q-LoRA

This repository contains a Jupyter Notebook (`baseline_2_t5_base.ipynb`) for fine-tuning a `T5-base` model on **zero-shot prompting** at the time of model building and at the time of inference.  
This process evaluates the inherent ability
 of large pre-trained models to understand and summarize
 complex, sensitive mental health conversations without
 task-specific training data.


---

## 📁 Model CheckPoint

All the model check point are found at - https://drive.google.com/drive/u/1/folders/1lh10LUhCJtEBqgsS_0_L_xTZ2YS3Muh-

```
├── Go to the link
├── Open and unzip `baseline_t5_llama_prompting.zip`          
├── Go to the Model Dir        
├── You can see the "Best Model" dir.        
├── For tokenizer load that using .pretrained(Model Dir) approch       
├── For Model load that also using .pretrained(Model Dir) approch                                             
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

## 👾 Prompting Example
```python
zero_shot_prompt = """[INST] <<SYS>>
 You are a helpful assistant specialized in
 summarizing mental health counseling
 dialogues. Generate a concise summary
 capturing the key points, topics
 discussed, and outcomes of the
 conversation.
 <</SYS>>
 Summarize the following dialogue:
 {dialogue} [/INST]
 Summary:"""
```

## ✨ Model Highlights

- **Zero-sort Prompting**: Wrote a generalizable prompt and pack that using all training and validation samples before training and after pass in the model for fine-tuning.
- **T5-base/MentalLLama**: Pretrained transformer model used for summarization
- **Q-LoRA + PEFT**: Efficient fine-tuning with quantization via Low-Rank Adaptation (LoRA) using Hugging Face's PEFT library
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
