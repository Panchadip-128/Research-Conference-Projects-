# Project Configuration

import torch

# Paths
RAW_DATA_PATH = '/kaggle/input/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv'
CHECKPOINT_DIR = './checkpoints'

# Data Processing
BATCH_SIZE = 32
GRAPH_BATCH_SIZE = 10
TRAIN_TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Training Hyperparameters
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
NUM_ROUNDS = 30
DRIFT_THRESHOLD = 0.05
POS_WEIGHT = torch.tensor([3.0]) 

# Scanning
CHUNK_SIZE = 200000
REQUIRED_SAMPLES = 2500
