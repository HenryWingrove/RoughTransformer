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


wandb.login()

#############################################
# Configuration Parameters for Rough Transformer
#############################################

WANDB_DEFAULT = False

# Training parameters
EPOCH = 10
BATCH_SIZE = 20
EVAL_BATCH_SIZE = -1

# Signature parameters – if ONLINE_SIGNATURE_CALC is False, signatures are precomputed offline
USE_SIGNATURES = False
ONLINE_SIGNATURE_CALC = False   # Offline precomputation; all computations occur on GPU
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
ADD_TIME = True     # Add a time channel (using epoch time)
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
INPUT_SIZE = 2 
SEQ_LEN = 100   # Length of each sequence window

CSV_FILE = '/kaggle/input/fb-2019-01-03/FB_2019-01-03_34200000_57600000_orderbook_10.csv'

# Device selection: We use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#############################################
# Raw Dataset for LOB Data
#############################################

class LOBDataset(Dataset):
    def __init__(self, csv_file, seq_len=SEQ_LEN):
        self.data = pd.read_csv(csv_file)
        # Convert DateTime to datetime and then to epoch seconds.
        self.data['DateTime'] = pd.to_datetime(self.data['DateTime'], format='%Y-%m-%d %H:%M:%S:%f')
        self.data['Timestamp'] = self.data['DateTime'].astype(np.int64) // 10**9
        self.data.sort_values('Timestamp', inplace=True)
        self.seq_len = seq_len


        self.feature_cols = [col for col in self.data.columns if col not in ['DateTime', 'Order_ID']]

        self.sequences = []
        self.labels = []
        for i in range(len(self.data) - seq_len):
            seq = self.data.iloc[i:i+seq_len]
            self.sequences.append(seq)
            # Use the "Price" column for label calculation.
            price_prev = self.data.iloc[i+seq_len-1]['Price']
            price_next = self.data.iloc[i+seq_len]['Price']
            label = 1 if price_next > price_prev else 0
            self.labels.append(label)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        # Convert features to a torch tensor (remains on CPU initially)
        x = torch.tensor(seq[self.feature_cols].values, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.long)
        return {'input': x, 'label': y}

#############################################
# Precomputed Dataset for Offline Signatures (on GPU)
#############################################

class PrecomputedLOBDataset(Dataset):
    """
    This dataset wraps a raw LOBDataset and precomputes the signature representation
    for each sample once (offline) on the GPU. The precomputed signature is then stored
    and used during training.
    """
    def __init__(self, raw_dataset, config, device):
        self.samples = []
        for i in range(len(raw_dataset)):
            sample = raw_dataset[i]
            x = sample['input']  # shape: (seq_len, raw_feature_dim)
            # If a time channel is to be added, create the time vector and concatenate:
            if config.add_time:
                t = (torch.linspace(0, config.seq_len, config.seq_len) / config.seq_len).reshape(-1, 1).to(device)
                x = torch.cat([t, x.to(device)], dim=1)
            else:
                x = x.to(device)
            # x now has shape (seq_len, raw_dim)
            # Create an x-axis for signature computation (e.g., equally spaced)
            x_axis = np.linspace(0, x.shape[0], x.shape[0])
            # Compute the signature on the GPU (make sure ComputeSignatures uses the passed device)
            # We add a batch dimension (size=1) and then remove it.
            sig = ComputeSignatures(x.unsqueeze(0), x_axis, config, device=device)
            sig = sig.squeeze(0)
            self.samples.append({'input': sig, 'label': sample['label']})
        if len(self.samples) > 0:
            self.sig_seq_len = self.samples[0]['input'].shape[0]
            self.sig_dim = self.samples[0]['input'].shape[1]
        else:
            self.sig_seq_len = config.seq_len
            self.sig_dim = 0

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

#############################################
# Dataset Preprocessing Function
#############################################

def get_dataset_preprocess(config, seed, device):
    csv_file = config.csv_file
    seq_len = config.seq_len
    # Create the raw dataset
    raw_dataset = LOBDataset(csv_file, seq_len)
    # For quick testing
    total_samples = len(raw_dataset)
    sample_size = max(1, int(0.01 * total_samples))
    raw_dataset = Subset(raw_dataset, list(range(sample_size)))
    
    # If offline signature computation is used, precompute signatures on the GPU.
    if not config.online_signature_calc:
        dataset = PrecomputedLOBDataset(raw_dataset, config, device)
        effective_seq_len = dataset.sig_seq_len
        effective_feature_dim = dataset.sig_dim
    else:
        dataset = raw_dataset
        effective_seq_len = config.seq_len
        effective_feature_dim = len(raw_dataset.dataset.feature_cols)
        if config.add_time:
            effective_feature_dim += 1

    # Split dataset into training, validation, and test sets.
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

    return train_loader, val_loader, test_loader, effective_seq_len, n_total, effective_feature_dim

#############################################
# Rough Transformer Training Code with Signature Patching
#############################################

def calculate_accuracy(config, model, data_loader, num_classes, seq_length, all_indices, indices_keep):
    model.eval()
    correct = 0
    total = 0
    error_distribution = {}
    # If adding time online, create time vector on the given device.
    if config.add_time:
        t = (torch.linspace(0, seq_length, seq_length) / seq_length).reshape(-1, 1).to(device)
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            if config.online_signature_calc:
                x_axis = np.linspace(0, inputs.shape[1], inputs.shape[1])
                x_axis = x_axis[indices_keep]
                if config.add_time:
                    inputs = torch.cat([t.repeat(inputs.shape[0], 1, 1), inputs.to(device)], dim=2)
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
    
    # Preprocess the dataset (using precomputed offline signatures if enabled)
    train_loader, val_loader, test_loader, seq_length_effective, n_total, effective_feature_dim = get_dataset_preprocess(config, seed, device)
    print(f"Effective sequence length: {seq_length_effective}")
    print(f"Effective feature dimension (input to model): {effective_feature_dim}")
    
    # For simplicity, we use all indices.
    all_indices = list(range(seq_length_effective))
    indices_keep = all_indices

    # Set input dimension for the model:
    # In online mode, the input dimension is computed from raw features;
    # In offline mode, we use the precomputed signature dimension.
    if config.online_signature_calc:
        raw_dim = len(LOBDataset(config.csv_file).feature_cols) + (1 if config.add_time else 0)
        signature_dim = raw_dim + raw_dim * raw_dim
        input_dim_for_model = signature_dim
    else:
        input_dim_for_model = effective_feature_dim

    print(f"Input dimension for model: {input_dim_for_model}")

    # Initialize the model (Transformer or LSTM)
    if config.model == 'transformer':
        model = DecoderTransformer(
            config,
            input_dim=input_dim_for_model,
            n_head=config.n_head,
            layer=config.num_layers,
            seq_num=n_total,
            n_embd=config.embedded_dim,
            win_len=seq_length_effective,
            num_classes=2
        ).to(device)
    elif config.model == 'lstm':
        model = LSTM_Classification(
            input_size=input_dim_for_model,
            hidden_size=10,
            num_layers=100,
            batch_first=True,
            num_classes=2
        ).to(device)
    else:
        raise ValueError('Model not supported')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    train_loss_list = []
    val_accuracy_list = []
    global_step = 0
    val_accuracy_best = -float('inf')
    epoch_start_time = time.time()
    
    # If adding time online, prepare the time vector on the chosen device.
    if config.add_time and config.online_signature_calc:
        t = (torch.linspace(0, seq_length_effective, seq_length_effective) / seq_length_effective).reshape(-1, 1).to(device)
    
    # Training loop
    for epoch in range(config.epoch):
        epoch_loss = 0.0
        model.train()
        for batch in train_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            if config.online_signature_calc:
                if config.add_time:
                    inputs = torch.cat([t.repeat(inputs.shape[0], 1, 1), inputs.to(device)], dim=2)
                x_axis = np.linspace(0, inputs.shape[1], inputs.shape[1])
                x_axis = x_axis[indices_keep]
                inputs = inputs[:, indices_keep, :]
                inputs = ComputeSignatures(inputs, x_axis, config, device)
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            global_step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        val_accuracy, _ = calculate_accuracy(config, model, val_loader, 2, seq_length_effective, all_indices, indices_keep)
        test_accuracy, _ = calculate_accuracy(config, model, test_loader, 2, seq_length_effective, all_indices, indices_keep)
        
        if val_accuracy > val_accuracy_best:
            val_accuracy_best = val_accuracy
            best_epoch = epoch
            test_at_best_epoch = test_accuracy
            # Save the best model checkpoint on GPU (it will be saved as a file that you can later load)
            torch.save(model.state_dict(), "rformer_best_model.pt")
            print(f"Best model saved at epoch {epoch+1} to rformer_best_model.pt")
        
        print(f"Epoch {epoch+1}/{config.epoch}, Loss: {avg_epoch_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%, Test Acc: {test_accuracy*100:.2f}%")
        val_accuracy_list.append(val_accuracy)
    
    training_time = time.time() - epoch_start_time
    total_time = time.time() - start_total_time
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    
    final_test_accuracy, _ = calculate_accuracy(config, model, test_loader, 2, seq_length_effective, all_indices, indices_keep)
    print(f"Final Test accuracy: {final_test_accuracy:.4f}")
    return final_test_accuracy

def main(config=None):
    # Optionally wrap in wandb
    if config.wandb:
        with wandb.init(project="RFormer", config=config, allow_val_change=True):
            config = wandb.config
            seed = 42
            torch.manual_seed(seed)
            print(f"Test accuracy: {train_transformer(config, seed)}")
    else:
        seed = 42
        torch.manual_seed(seed)
        print(f"Test accuracy: {train_transformer(config, seed)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epoch", type=int, default=EPOCH, help="Number of epochs to train the model")
    parser.add_argument("--use_signatures", action='store_true' if USE_SIGNATURES else 'store_false', default=USE_SIGNATURES, help="Whether to use signatures")
    parser.add_argument("--sig_win_len", type=int, default=SIG_WIN_LEN, help="Length of the signature window")
    parser.add_argument("--sig_level", type=int, default=SIG_LEVEL, help="Level of the signature")
    parser.add_argument("--num_windows", type=int, default=NUM_WINDOWS, help="Number of windows")
    parser.add_argument("--univariate", action='store_true' if UNIVARIATE else 'store_false', default=UNIVARIATE, help="Use univariate data")
    parser.add_argument("--irreg", action='store_true' if IRREG else 'store_false', default=IRREG, help="Use irregular intervals")
    parser.add_argument("--add_time", action='store_true' if ADD_TIME else 'store_false', default=ADD_TIME, help="Add a time channel")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS, help="Number of random seeds")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Dataset identifier")
    parser.add_argument("--csv_file", type=str, default=CSV_FILE, help="Path to CSV file")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, help="Sequence length")
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE, help="Input size (raw features)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--n_head", type=int, default=N_HEAD, help="Number of heads in the transformer")
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS, help="Number of layers")
    parser.add_argument("--epochs_for_convergence", type=int, default=EPOCHS_FOR_CONVERGENCE, help="Epochs for convergence")
    parser.add_argument("--accuracy_for_convergence", type=float, default=ACCURACY_FOR_CONVERGENCE, help="Accuracy threshold")
    parser.add_argument("--std_for_convergence", type=float, default=STD_FOR_CONVERGENCE, help="Standard deviation threshold")
    parser.add_argument("--embedded_dim", type=int, default=EMBEDDED_DIM, help="Dimension of embedded features")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--embd_pdrop", type=float, default=EMBD_PDROP, help="Dropout probability for embeddings")
    parser.add_argument("--attn_pdrop", type=float, default=ATTN_PDROP, help="Dropout probability for attention")
    parser.add_argument("--resid_pdrop", type=float, default=RESID_PDROP, help="Dropout probability for residual connections")
    parser.add_argument("--overlap", action='store_true' if OVERLAP else 'store_false', default=OVERLAP, help="Whether to overlap data")
    parser.add_argument("--scale_att", action='store_true' if SCALE_ATT else 'store_false', default=SCALE_ATT, help="Scale attention weights")
    parser.add_argument("--sparse", action='store_true' if SPARSE else 'store_false', default=SPARSE, help="Use sparse connections")
    parser.add_argument("--v_partition", type=float, default=V_PARTITION, help="Validation partition ratio")
    parser.add_argument("--q_len", type=int, default=Q_LEN, help="Length of the query")
    parser.add_argument("--early_stop_ep", type=int, default=EARLY_STOP_EP, help="Early stopping epoch")
    parser.add_argument("--sub_len", type=int, default=SUB_LEN, help="Subsequence length")
    parser.add_argument("--warmup_proportion", type=float, default=WARMUP_PROPORTION, help="Warmup proportion")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="Optimizer")
    parser.add_argument("--continue_training", action='store_true' if CONTINUE_TRAINING else 'store_false', default=CONTINUE_TRAINING, help="Continue training")
    parser.add_argument("--save_all_epochs", action='store_true' if SAVE_ALL_EPOCHS else 'store_false', default=SAVE_ALL_EPOCHS, help="Save model every epoch")
    parser.add_argument("--pretrained_model_path", type=str, default=PRETRAINED_MODEL_PATH, help="Pretrained model path")
    parser.add_argument("--downsampling", action='store_true' if DOWNSAMPLING else 'store_false', default=DOWNSAMPLING, help="Downsample data")
    parser.add_argument("--zero_shot_downsample", action='store_true' if ZERO_SHOT_DOWNSAMPLE else 'store_false', default=ZERO_SHOT_DOWNSAMPLE, help="Zero-shot downsample")
    parser.add_argument("--use_random_drop", action='store_true' if USE_RANDOM_DROP else 'store_false', default=USE_RANDOM_DROP, help="Use random drop")
    parser.add_argument("--random_percentage", type=float, default=RANDOM_PERCENTAGE, help="Random drop percentage")
    parser.add_argument("--model", type=str, default=MODEL, help="Model type (transformer or lstm)")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Test set ratio")
    parser.add_argument("--val_size", type=float, default=VAL_SIZE, help="Validation set ratio")
    parser.add_argument("--global_backward", action='store_true' if GLOBAL_BACKWARD else 'store_false', default=GLOBAL_BACKWARD, help="Use global backward signatures")
    parser.add_argument("--global_forward", action='store_true' if GLOBAL_FORWARD else 'store_false', default=GLOBAL_FORWARD, help="Use global forward signatures")
    parser.add_argument("--local_tight", action='store_true' if LOCAL_TIGHT else 'store_false', default=LOCAL_TIGHT, help="Use local tight signatures")
    parser.add_argument("--local_wide", action='store_true' if LOCAL_WIDE else 'store_false', default=LOCAL_WIDE, help="Use local wide signatures")
    parser.add_argument("--local_width", type=float, default=LOCAL_WIDTH, help="Local window width")
    parser.add_argument("--online_signature_calc", action='store_true' if ONLINE_SIGNATURE_CALC else 'store_false', default=ONLINE_SIGNATURE_CALC, help="Use online signature calculation")
    parser.add_argument("--wandb", action='store_true' if WANDB_DEFAULT else 'store_false', default=WANDB_DEFAULT, help="Use wandb logging")
    
    config, _ = parser.parse_known_args()
    main(config)