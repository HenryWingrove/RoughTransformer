import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import random
import argparse
import wandb
import time
import statistics
import pprint
import types

# Import your custom model definitions.
from model_classification import DecoderTransformer  # Rough Transformer variant
from lstm_classification import LSTM_Classification  # Alternative model (if needed)
from sig_utils import ComputeSignatures  # Your signature computation function

# Optionally log in to wandb
wandb.login()

#############################################
# Configuration Parameters for Rough Transformer
#############################################

WANDB_DEFAULT = False

# Training parameters
EPOCH = 10
BATCH_SIZE = 20
EVAL_BATCH_SIZE = -1

# Signature parameters – enable signature pathing for asynchronous LOB data
USE_SIGNATURES = True
ONLINE_SIGNATURE_CALC = False   # When False, use offline precomputed signatures
SIG_WIN_LEN = 50
SIG_LEVEL = 2
NUM_WINDOWS = 100

UNIVARIATE = False  # LOB/trade data is multivariate

# Options for multi-view signature attention
GLOBAL_BACKWARD = True
GLOBAL_FORWARD = False
LOCAL_TIGHT = False
LOCAL_WIDE = False
LOCAL_WIDTH = 50

IRREG = True
ADD_TIME = True     # Add a time channel
N_SEEDS = 1
DATASET = 'LOB_Dataset'

N_HEAD = 3
NUM_LAYERS = 1
EPOCHS_FOR_CONVERGENCE = 10000
ACCURACY_FOR_CONVERGENCE = 0.6
STD_FOR_CONVERGENCE = 0.05
EMBEDDED_DIM = 10
LR = 0.00040788
WEIGHT_DECAY = 0
EMBD_PDROP = 0.1
ATTN_PDROP = 0.1
RESID_PDROP = 0.1
OVERLAP = False
SCALE_ATT = False
SPARSE = False
V_PARTITION = 0.1
Q_LEN = 1
EARLY_STOP_EP = 500
SUB_LEN = 1
WARMUP_PROPORTION = -1
OPTIMIZER = 'Adam'
CONTINUE_TRAINING = False
SAVE_ALL_EPOCHS = False
PRETRAINED_MODEL_PATH = ''
DOWNSAMPLING = False
ZERO_SHOT_DOWNSAMPLE = False
USE_RANDOM_DROP = False
RANDOM_PERCENTAGE = 0.7
MODEL = 'transformer'  # or 'lstm'
TEST_SIZE = 0.3
VAL_SIZE = 0.5
INPUT_SIZE = 2  # (number of raw features in CSV)
SEQ_LEN = 100   # Length of each sequence window

# CSV file location
CSV_FILE = '/Users/henrywingrove/Documents/Projects_/RFormer/RFormer/src/UEA/FB_2019-01-03_34200000_57600000_orderbook_10.csv'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

#############################################
# Custom Dataset for LOB and Trade Data
#############################################

class LOBDataset(Dataset):
    def __init__(self, csv_file, seq_len=SEQ_LEN):
        self.data = pd.read_csv(csv_file)
        self.data['DateTime'] = pd.to_datetime(self.data['DateTime'], format='%Y-%m-%d %H:%M:%S:%f')
        self.data['Timestamp'] = self.data['DateTime'].astype(np.int64) // 10**9
        self.data.sort_values('Timestamp', inplace=True)
        self.seq_len = seq_len
        # Initially, use all columns except DateTime and Order_ID as features.
        self.feature_cols = [col for col in self.data.columns if col not in ['DateTime', 'Order_ID', 'DateTime', 'Event_Type', 'Order_ID', 'Size', 'Price', 'Direction']]
        self.sequences = []
        self.labels = []
        for i in range(len(self.data) - seq_len):
            seq = self.data.iloc[i:i+seq_len]
            self.sequences.append(seq)
            price_prev = self.data.iloc[i+seq_len-1]['Price']
            price_next = self.data.iloc[i+seq_len]['Price']
            label = 1 if price_next > price_prev else 0
            self.labels.append(label)
        # For offline signature mode, we will store precomputed signatures here.
        self.precomputed_inputs = None

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # If offline signatures were precomputed, return them.
        if self.precomputed_inputs is not None:
            x = self.precomputed_inputs[idx]
        else:
            seq = self.sequences[idx]
            x = torch.tensor(seq[self.feature_cols].values, dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'input': x, 'label': y}

def get_dataset_preprocess(config, seed, device):
    csv_file = config.csv_file
    seq_len = config.seq_len
    dataset = LOBDataset(csv_file, seq_len)
    
    # For quick testing, take a small subset (e.g., 1% of samples)
    total_samples = len(dataset)
    sample_size = max(1, int(0.0001 * total_samples))
    dataset = Subset(dataset, list(range(sample_size)))
    
    # If using offline signatures, precompute them once now.
    if USE_SIGNATURES and not ONLINE_SIGNATURE_CALC:
        print("Precomputing offline signatures for the dataset...")
        precomputed = []
        # Compute raw dimension (include time if ADD_TIME is True)
        raw_dim = len(dataset.dataset.feature_cols) + (1 if config.add_time else 0)
        # For depth 2, expected signature dimension:
        signature_dim = raw_dim + raw_dim * raw_dim
        for i in range(len(dataset)):
            sample = dataset.dataset.__getitem__(i)  # get raw sample from the underlying dataset
            x = sample['input']  # shape: (seq_len, num_raw_features)
            if config.add_time:
                t = (torch.linspace(0, config.seq_len, config.seq_len) / config.seq_len).reshape(-1, 1).to(device)
                x = torch.cat([t, x], dim=1)
            x = x.unsqueeze(0)  # shape: (1, seq_len, raw_dim)
            # Use all time indices (we could also use a subset if desired)
            x_axis = np.linspace(0, config.seq_len, config.seq_len)
            # Compute signatures (offline) – note: ComputeSignatures should be defined to work on batch inputs.
            sig = ComputeSignatures(x, x_axis, config, device)  # expected shape: (1, T_sig, signature_dim)
            sig = sig.squeeze(0)  # shape: (T_sig, signature_dim)
            precomputed.append(sig)
        # Cache the precomputed offline signatures in the dataset.
        dataset.dataset.precomputed_inputs = precomputed
        print("Offline signatures precomputed.")
    
    # Split dataset indices into train, val, test.
    n_total = len(dataset)
    n_test = int(n_total * config.test_size)
    n_val = int(n_total * config.val_size)
    n_train = n_total - n_test - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n_total))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    seq_length_original = config.seq_len
    num_classes = 2
    num_samples = len(dataset)
    # For offline mode, effective feature dimension is the precomputed signature dimension.
    if USE_SIGNATURES and not ONLINE_SIGNATURE_CALC:
        num_features_original = precomputed[0].shape[-1]
    else:
        num_features_original = len(dataset.dataset.feature_cols)
    return train_loader, val_loader, test_loader, seq_length_original, num_classes, num_samples, num_features_original

#############################################
# Rough Transformer Training Code with Signature Patching
#############################################

def calculate_accuracy(config, model, data_loader, num_classes, seq_length_original, all_indices, indices_keep):
    model.eval()
    correct = 0
    total = 0
    error_distribution = {}
    if config.add_time:
        t = (torch.linspace(0, seq_length_original, seq_length_original) / seq_length_original).reshape(-1, 1).to(device)
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            # In offline mode, the inputs are already precomputed.
            if config.online_signature_calc:
                x_axis = np.linspace(0, inputs.shape[1], inputs.shape[1])
                x_axis = x_axis[indices_keep]
                if config.add_time:
                    inputs = torch.cat([t.repeat(inputs.shape[0], 1, 1), inputs], dim=2)
                inputs = inputs[:, indices_keep, :]
                inputs = ComputeSignatures(inputs, x_axis, config, device)
            outputs = model(inputs).to(device)
            predicted_labels = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
            incorrect_labels = labels[predicted_labels != labels]
            if incorrect_labels.numel() > 0:
                unique_labels = torch.unique(incorrect_labels)
                error_counts = torch.bincount(incorrect_labels, minlength=num_classes)
                error_distribution = {label.item(): count.item() for label, count in zip(unique_labels, error_counts) if count > 0}
    accuracy = correct / total
    return accuracy, error_distribution

def train_transformer(config, seed):
    start_total_time = time.time()
    print("############### Config ###############")
    pprint.pprint(config.__dict__)
    print(f"\nSeed: {seed}\n")
    print("######################################")
    
    train_loader, val_loader, test_loader, seq_length_original, num_classes, num_samples, num_features_original = get_dataset_preprocess(config, seed, device)
    print(f'Number of classes: {num_classes}')
    print(f'Raw feature count from CSV: {num_features_original}')
    
    converged = False
    all_indices = list(range(seq_length_original))
    indices_keep = all_indices  # (no random drop in this example)
    
    # Compute effective input dimension:
    if USE_SIGNATURES:
        if config.online_signature_calc:
            raw_dim = len(train_loader.dataset.dataset.feature_cols) + (1 if config.add_time else 0)
            signature_dim = raw_dim + raw_dim * raw_dim
            input_dim_for_model = signature_dim
        else:
            # In offline mode, the dataset now returns precomputed signatures.
            input_dim_for_model = num_features_original
    else:
        input_dim_for_model = len(train_loader.dataset.dataset.feature_cols)
    
    print(f"Effective input dimension for model (passed to TransformerModel): {input_dim_for_model}")
    
    if config.model == 'transformer':
        model = DecoderTransformer(
            config,
            input_dim=input_dim_for_model,
            n_head=config.n_head,
            layer=config.num_layers,
            seq_num=num_samples,
            n_embd=config.embedded_dim,
            win_len=seq_length_original,
            num_classes=num_classes
        ).to(device)
    elif config.model == 'lstm':
        model = LSTM_Classification(
            input_size=input_dim_for_model,
            hidden_size=10,
            num_layers=100,
            batch_first=True,
            num_classes=num_classes
        ).to(device)
    else:
        raise ValueError('Model not supported')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    train_loss_list = []
    val_accuracy_list = []
    global_step = 0
    val_accuracy_best = -float('inf')
    start_time = time.time()
    
    if config.add_time:
        t = (torch.linspace(0, seq_length_original, seq_length_original) / seq_length_original).reshape(-1, 1).to(device)
    
    # Training loop
    for epoch in range(config.epoch):
        epoch_loss = 0.0
        for idx, batch in enumerate(train_loader):
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            # In online mode, compute signatures on the fly.
            if config.online_signature_calc:
                if config.add_time:
                    inputs = torch.cat([t.repeat(inputs.shape[0], 1, 1), inputs], dim=2)
                x_axis = np.linspace(0, inputs.shape[1], inputs.shape[1])
                x_axis = x_axis[indices_keep]
                inputs = inputs[:, indices_keep, :]
                inputs = ComputeSignatures(inputs, x_axis, config, device)
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            global_step += 1
            train_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        if config.wandb:
            wandb.log({'losses/avg_epoch_loss': avg_epoch_loss})
        
        val_accuracy, _ = calculate_accuracy(config, model, val_loader, num_classes, seq_length_original, all_indices, indices_keep)
        test_accuracy, _ = calculate_accuracy(config, model, test_loader, num_classes, seq_length_original, all_indices, indices_keep)
        
        if config.wandb:
            wandb.log({'accuracies/validation_accuracy': val_accuracy,
                       'accuracies/test_accuracy': test_accuracy})
        model.train()
        
        if len(val_accuracy_list) > config.epochs_for_convergence and not converged:
            avg_val = statistics.mean(val_accuracy_list[-config.epochs_for_convergence:])
            std_val = statistics.stdev(val_accuracy_list[-config.epochs_for_convergence:])
            if avg_val > config.accuracy_for_convergence and std_val <= config.std_for_convergence * avg_val:
                convergence_time = time.time() - start_time
                print(f'Model has converged at epoch {epoch}. Time taken: {convergence_time:.2f} seconds.')
                converged = True
        
        if val_accuracy > val_accuracy_best:
            val_accuracy_best = val_accuracy
            best_epoch = epoch
            test_at_best_epoch = test_accuracy
            torch.save(model.state_dict(), "rformer_best_model.pt")
            print(f"Best model saved at epoch {epoch+1} to rformer_best_model.pt")
        
        print(f'Epoch {epoch+1}/{config.epoch}, Loss: {loss.item():.4f}, ' +
              f'Valid Accuracy: {val_accuracy*100:.2f}%, Best Valid Accuracy: {val_accuracy_best*100:.2f}%, ' +
              f'Test Accuracy: {test_accuracy*100:.2f}%')
        val_accuracy_list.append(val_accuracy)
    
    execution_time = time.time() - start_time
    total_time = time.time() - start_total_time
    print(f"Training time: {execution_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    
    test_accuracy, test_error_distribution = calculate_accuracy(config, model, test_loader, num_classes, seq_length_original, all_indices, indices_keep)
    if config.wandb:
        wandb.log({'performance/best_val_accuracy': val_accuracy_best,
                   'performance/test_accuracy_at_end': test_accuracy,
                   'performance/best_epoch': best_epoch,
                   'performance/test_at_best_epoch': test_at_best_epoch})
    return test_accuracy

def main(config=None):
    if config.wandb:
        with wandb.init(project="RFormer", config=config, allow_val_change=True):
            config = wandb.config
            if config.n_seeds == 1:
                seed = 42
                torch.manual_seed(seed)
                print(f"Test accuracy: {train_transformer(config, seed)}")
            else:
                results = []
                for seed in range(config.n_seeds):
                    torch.manual_seed(42 + seed)
                    results.append(train_transformer(config, 42 + seed))
                    print(results[-1])
                results = np.array(results)
                print(f"Average accuracy: {results.mean()}, Std: {results.std()}")
                return results.mean(), results.std()
    else:
        if config.n_seeds == 1:
            seed = 42
            torch.manual_seed(seed)
            print(f"Test accuracy: {train_transformer(config, seed)}")
        else:
            results = []
            for seed in range(config.n_seeds):
                torch.manual_seed(42 + seed)
                results.append(train_transformer(config, 42 + seed))
                print(results[-1])
            results = np.array(results)
            print(f"Average accuracy: {results.mean()}, Std: {results.std()}")
            return results.mean(), results.std()

#############################################
# Argument Parser and Entry Point
#############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epoch", type=int, default=EPOCH, help="Number of epochs to train the model")
    parser.add_argument("--use_signatures", action='store_true' if USE_SIGNATURES else 'store_false', default=USE_SIGNATURES, help="Whether to use signatures in the model")
    parser.add_argument("--sig_win_len", type=int, default=SIG_WIN_LEN, help="Length of the signature window")
    parser.add_argument("--sig_level", type=int, default=SIG_LEVEL, help="Level of the signature")
    parser.add_argument("--num_windows", type=int, default=NUM_WINDOWS, help="Number of windows to use")
    parser.add_argument("--univariate", action='store_true' if UNIVARIATE else 'store_false', default=UNIVARIATE, help="Use univariate data")
    parser.add_argument("--irreg", action='store_true' if IRREG else 'store_false', default=IRREG, help="Use irregular intervals")
    parser.add_argument("--add_time", action='store_true' if ADD_TIME else 'store_false', default=ADD_TIME, help="Add a time channel to each input")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS, help="Number of random seeds to use")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Dataset identifier")
    parser.add_argument("--csv_file", type=str, default=CSV_FILE, help="Path to CSV file with LOB and trade data")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, help="Sequence length (number of events per sample)")
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE, help="Input size (number of features)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--n_head", type=int, default=N_HEAD, help="Number of heads in the transformer model")
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS, help="Number of layers in the model")
    parser.add_argument("--epochs_for_convergence", type=int, default=EPOCHS_FOR_CONVERGENCE, help="Epochs required for convergence")
    parser.add_argument("--accuracy_for_convergence", type=float, default=ACCURACY_FOR_CONVERGENCE, help="Accuracy threshold for convergence")
    parser.add_argument("--std_for_convergence", type=float, default=STD_FOR_CONVERGENCE, help="Standard deviation threshold for convergence")
    parser.add_argument("--embedded_dim", type=int, default=EMBEDDED_DIM, help="Dimension of embedded features")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay for the optimizer")
    parser.add_argument("--embd_pdrop", type=float, default=EMBD_PDROP, help="Dropout probability for embeddings")
    parser.add_argument("--attn_pdrop", type=float, default=ATTN_PDROP, help="Dropout probability for attention")
    parser.add_argument("--resid_pdrop", type=float, default=RESID_PDROP, help="Dropout probability for residual connections")
    parser.add_argument("--overlap", action='store_true' if OVERLAP else 'store_false', default=OVERLAP, help="Whether to overlap data")
    parser.add_argument("--scale_att", action='store_true' if SCALE_ATT else 'store_false', default=SCALE_ATT, help="Whether to scale attention weights")
    parser.add_argument("--sparse", action='store_true' if SPARSE else 'store_false', default=SPARSE, help="Use sparse connections")
    parser.add_argument("--v_partition", type=float, default=V_PARTITION, help="Partition ratio for validation")
    parser.add_argument("--q_len", type=int, default=Q_LEN, help="Length of the query in the model")
    parser.add_argument("--early_stop_ep", type=int, default=EARLY_STOP_EP, help="Early stopping after this many epochs")
    parser.add_argument("--sub_len", type=int, default=SUB_LEN, help="Length of the subsequences")
    parser.add_argument("--warmup_proportion", type=float, default=WARMUP_PROPORTION, help="Warmup proportion for learning rate schedule")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="Optimizer to use")
    parser.add_argument("--continue_training", action='store_true' if CONTINUE_TRAINING else 'store_false', default=CONTINUE_TRAINING, help="Continue training from a checkpoint")
    parser.add_argument("--save_all_epochs", action='store_true' if SAVE_ALL_EPOCHS else 'store_false', default=SAVE_ALL_EPOCHS, help="Save model after every epoch")
    parser.add_argument("--pretrained_model_path", type=str, default=PRETRAINED_MODEL_PATH, help="Path to pretrained model")
    parser.add_argument("--downsampling", action='store_true' if DOWNSAMPLING else 'store_false', default=DOWNSAMPLING, help="Whether to downsample the data")
    parser.add_argument("--zero_shot_downsample", action='store_true' if ZERO_SHOT_DOWNSAMPLE else 'store_false', default=ZERO_SHOT_DOWNSAMPLE, help="Apply zero-shot downsampling")
    parser.add_argument("--use_random_drop", action='store_true' if USE_RANDOM_DROP else 'store_false', default=USE_RANDOM_DROP, help="Use random sampling")
    parser.add_argument("--random_percentage", type=float, default=RANDOM_PERCENTAGE, help="Percentage of data to sample randomly")
    parser.add_argument("--model", type=str, default=MODEL, help="Model type to use (transformer or lstm)")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Percentage of data reserved for testing")
    parser.add_argument("--val_size", type=float, default=VAL_SIZE, help="Percentage of test set reserved for validation")
    
    parser.add_argument("--global_backward", action='store_true' if GLOBAL_BACKWARD else 'store_false', default=GLOBAL_BACKWARD, help="Use global backward signatures")
    parser.add_argument("--global_forward", action='store_true' if GLOBAL_FORWARD else 'store_false', default=GLOBAL_FORWARD, help="Use global forward signatures")
    parser.add_argument("--local_tight", action='store_true' if LOCAL_TIGHT else 'store_false', default=LOCAL_TIGHT, help="Use local tight signatures")
    parser.add_argument("--local_wide", action='store_true' if LOCAL_WIDE else 'store_false', default=LOCAL_WIDE, help="Use local wide signatures")
    parser.add_argument("--local_width", type=float, default=LOCAL_WIDTH, help="Width of local window")
    parser.add_argument("--online_signature_calc", action='store_true' if ONLINE_SIGNATURE_CALC else 'store_false', default=ONLINE_SIGNATURE_CALC, help="Use online signature calculation")
    parser.add_argument("--wandb", action='store_true' if WANDB_DEFAULT else 'store_false', default=WANDB_DEFAULT, help="Use wandb for logging")
    
    # Use parse_known_args() to ignore extra Jupyter/Kaggle kernel arguments
    config, _ = parser.parse_known_args()
    main(config)