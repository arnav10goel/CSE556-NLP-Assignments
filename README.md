# CSE556-NLP-Assignments
This repository contains the course assignments given for the course Natural Language Processing offered in Winter'24. There were 4 assignments and the an overview for each is given as follows:

### 1. Assignment-1: 
In this assignment, we had to implement a `Byte-Pair Encoder (BPE)` Tokeniser and a Bigram Language Model for generating emotive speech along with smoothing techniques such as `Kneser-ney Smoothing`.

### 2. Assignment-2: 
In this assignment, we had to perform a `Named Entity Recognition (NER` and an `Aspect Term Extraction` task. The two tasks are modelled as sequence labelling tasks using BIO encoding. We implemented a RNN, LSTM and GRU-based model as baselines and implemented a BiLSTM-CRF based model for improvement over the baseline.

### 3. Assignment-3:
In this assignment, we had to perform 2 tasks:
1. `Sentence Similarity Task (STS)`: Given two sentences we had to predict a similarity score for the two. We first fine-tuned a BERT model on this task. Subsequent work involved evaluating the sentence-transformers library in a zero-shot fashion and then fine-tuning it on the downstream task using `CosineSimilarityLoss`.
2. `Machine Translation (MT)`: For this task we had to train models for a German to English translation task. We implemented a `Transformer` model from scratch in PyTorch. We additionally fine-tune a `T5-small` model on this task.

### 4. Assignment-4:
In this assignment, we prepared and benchmarked-transformer based models on 2 tasks i.e. Emotion Recognition in Conversation (ERC) and Emotion Flip Reasoning (EFR).
