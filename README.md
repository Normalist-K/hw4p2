# Instructions for HW4P2 (Youngin Kim, youngin2)
11-785 HW4P2: Attention-based Speech Recognition

## 0. Prerquisite

**Please modify `configs/path/server.yaml` for your own path**


## 1. Best Architecture

**You can see Hyperparameters values in `configs/`.**

**Or You can see the details and experimental results in [WanDB](https://wandb.ai/normalkim/hw4p2?workspace=user-normalkim).**

    - best model: cnn_clipping-07:16:09:19
  
- Model Architecture
    - You can check in `src/models/seq2seq.py`
    - Based on LAS architecture
    - Added cnn embedding in encoder
- Optimizer 
    - Adam
    - scheduler: CosineAnnealing
- Regularize
    - Locked Dropout for LSTM and pbLSTM layers
    - Dropout for embedding cnn
    - gradient clipping
    - teacher forcing and scheduling along with the learning rate
- Data & Augmentation
    - Cepstral Mean Normalize

## 2. Run
```
$ python run.py save_name={name_for_submission&weight_file}
```
