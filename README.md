# BERT-Transformer-Pytorch
Basic implementation of BERT and Transformer in Pytorch in one python file of ~300 lines of code (train.py).  
  
This project aims to provide an easy-to-run easy-to-understand code for NLP beginners and people who want to know how Transformers work.  
The project uses a simplified implementation of BERT (no labels are required for training).  
The original implementation of Transformer uses an encoder and a decoder, here we only need the encoder.  
The model can train in 30 minutes on 1 x RTX2070Super GPU.  
  
Visualization of word embeddings:
![alt text](https://miro.medium.com/max/3000/1*tyabpnOIHPhl1ZoQQuSfvw.png)


Implementation details: https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7

## "Predict next word" task

**August 2023 update**: 
- For experiment purposes, I also implemented the "predict next word" task which is used to train GPT.
- The code can be found in "main_predictnextword.py"
- This code is a slight modification of main.py
