# CS839-Project

# Project Due: Dec 13th!!!

## Dataset
- Wiki Text
- Model size: ~10M, 125M to 1B

## Work Division
- Build from scratch, nano model: Jingyun, Zhiqi*, Xinta
- Continue pretrain / Larger model (may fail): Ruida, Yikai*, Pan Jin

## Dream Training Concept
Original training process: the model is fed sequentially with training blocks {b_1, b_2, …, b_n}

We aim to mimic the human "dream" process:

### Training Steps and Batch Size
- Assume there are S total training steps, with n data records.
- Batch size B = S / n.
- The training is divided into three phases:
  1. The first 1/3 of S steps: normal training stage
  2. The middle 1/3 of S steps: "dream" training stage
  3. The last 1/3 of S steps: back to normal training

### Dream Training Process
- Organize the training into blocks of 100 steps.
- Based on psychology, about 10% of brain activity is REM sleep (dream sleep).
- To simulate dreaming: In each block of 100 steps, the first 90 steps use normal training data, while the last 10 steps use "dream" data.

### Dream Data Composition
- Randomly shuffle the previous 90 steps into 5 steps of training data (reorganizing seen data).
- Use the current model checkpoint to generate another 5 steps of training data (self-generated data by the model).
- Merge these two parts together to form 10 steps of "dream" training data.
