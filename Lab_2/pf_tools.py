import os 
from pathlib import Path 
import pickle
import pandas as pd
import warnings

import torch
from torch import nn, optim
import numpy as np
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _extract_tar

import librosa
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import librosa.display

import datetime
import csv

class CheckThisCell(Exception):
    pass

class SLPdata(Dataset):

    RELEASE_CONFIGS = {'train_small': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf26/lab2/train_small.tgz' , 'checksum':'f25ce94bd4e85a77285a589509d20f6264d173a819ada8077f03c03c3f755ef4'},
                'train': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf26/lab2/train.tgz' , 'checksum':'a5f3cb33c1a85c0d3ad099daf8928fa3af9e04ddfd6af88f71fdb1f475610384'},
                'dev': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf26/lab2/dev.tgz' , 'checksum':'84dbe6d5508d681b7a45543c3464e5c76015a53bed6b01b14e6c47ac56aa2522'},
                'evl': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf26/lab2/evl.tgz' , 'checksum':'2e135e4e56054be50c2673d4e4d3a5d7651222aceea6d0cb9038ae044edd8e53'}
                }
                   
    def __init__(self, root : str, dataset_id: str, transform_id: str = "feat", audio_transform : callable = None, chunk_size : int = -1, chunk_hop : int = -1, chunk_transform : callable = None) -> None:
        
        if dataset_id not in SLPdata.RELEASE_CONFIGS:
            raise ValueError("Not known data set in SLPdata")
        
        if audio_transform is None:
            raise ValueError("Need to define some tranformation from audiofile to features")

        self.path = Path(root) / dataset_id
        self.url = SLPdata.RELEASE_CONFIGS[dataset_id]['url']

        self.archive = os.path.basename(self.url)
        self.archive = Path(root) / self.archive

        self.audio_dir = self.path / 'wav'
        self.feat_dir = self.path / transform_id
        self.info_file = self.path / 'info.csv'

        self.audio_transform = audio_transform
        self.chunk_size = chunk_size
        self.chunk_hop = chunk_hop
        self.chunk_transform = chunk_transform

        if self.chunk_size > 0 and self.chunk_hop <= 0:
            self.chunk_hop = self.chunk_size

        self.download_data(SLPdata.RELEASE_CONFIGS[dataset_id]['checksum'])
        self.data_to_feat()

        if not os.path.isfile(self.info_file):
            raise RuntimeError("CSV file does not exist. There was some problem downloading data.")

        if not os.path.isdir(self.feat_dir):
            raise RuntimeError("Features directory does not exist. There was some problem applying feature extraction.")
    
    
        df = pd.read_csv(self.info_file)
        required_columns = {"wav", "gender", "age"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

        self._walker = []
        for _, row in df.iterrows():
            basename = row["wav"].split('.')[0].strip() 
            gender = row["gender"]
            age = row["age"] if row["age"] == "?" else float(row["age"])
                
            self._walker.extend([(c, basename, gender, age) for c in os.listdir(self.feat_dir / basename )])
    
    def __getitem__(self, index):
        featIn, basename, gender, age = self._walker[index]

        # feats = pickle.load(open(self.feat_dir / basename / featIn, 'rb'))
        feats = np.load(self.feat_dir / basename / featIn, allow_pickle=True)
        
        return feats, (gender, age), basename
    
    def __len__(self):
        return len(self._walker)
                   
    def download_data(self,  checksum : str = None) -> None:   
        if not os.path.isdir(self.path):
            if not os.path.isfile(self.archive):
                download_url_to_file(self.url, self.archive, hash_prefix=checksum)
            _extract_tar(self.archive)

    def data_to_feat(self) -> None:
        
        if not os.path.isfile(self.info_file):
            raise RuntimeError("CSV file does not exist. Please download it first. There was some problem during data download and extreaction.")

        if os.path.isdir(self.feat_dir):
            warnings.warn("The feature directory already exists, and no new feature extraction will be performed.")
        else:
            if not os.path.isdir(self.audio_dir):
                raise RuntimeError("Audios directory does not exist. There was some problem during data download and extreaction.")
   
            ## Create feature directory
            os.mkdir(self.feat_dir)

            # Extract features 
            df = pd.read_csv(self.info_file)
        
            for _, row in tqdm(df.iterrows()):
                basename = row["wav"].split('.')[0].strip() 
                audioin = self.audio_dir / f'{basename}.wav'
                featOutPath = self.feat_dir / basename
                
                if not os.path.isdir(featOutPath):
                    os.mkdir(featOutPath)
                
                # print(f'\t{audioin}...', end='')
                feats = self.audio_transform(str(audioin)) # The output of this is (Ntime x Dim)

                finish = False
                if self.chunk_size > 0:
                    for b in range(0, feats.shape[0], self.chunk_hop-1):
                        this_feats = feats[b:b+self.chunk_size]
                        if this_feats.shape[0] < self.chunk_size:
                            this_feats = np.concatenate((this_feats, np.zeros((self.chunk_size-this_feats.shape[0], this_feats.shape[1]), dtype=this_feats.dtype)))
                            finish = True
                        if self.chunk_transform is not None:
                            this_feats = self.chunk_transform(this_feats)

                        # pickle.dump(this_feats, open(featOutPath / f'{basename}.{b//self.chunk_hop}.feat' , 'wb'))
                        np.save(featOutPath / f'{basename}.{b//self.chunk_hop}.npy' , this_feats)
        
                        if finish:
                            break 
                else:
                    np.save(featOutPath / f'{basename}.npy' , feats)
                    # pickle.dump(feats, open(featOutPath / f'{basename}.feat' , 'wb'))
                        
## Auxiliary functions

# Plot audio waveform and spectrogram
def audioplot(filename, sr=16000, mono=True, duration=None):

    x, sr = librosa.load(filename, sr=sr, mono=mono, duration=duration)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
    
    librosa.display.waveshow(x, sr=sr,  ax=ax[0])
    ax[0].set(title='Waveform')
    ax[0].label_outer()
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax[1])
    ax[1].set(title='Linear-frequency power spectrogram')
    ax[1].label_outer()     
    
    return 

def prepare_slp_data(slp_data, collapse_samples=True, expand_labels=True):
    """
    This function permits preparing SLP data for both training and prediction:
    
        - `collapse_samples` permits concanating all the slp_data in a single dictionary with fields 
        'data' and 'label' containing all the data and labels respectively. If False, it returns a 
        dictionary with keys the identifer of esch feature file and with value a 'data' and 'label' 
        dictionary.
        
        - expand_labels permits to expand the labels with the same size of the corresponding data. 
        If False, the labels are kept as they are (one per feature file). Only used if collapse_samples is False.

    """
    
    if collapse_samples:
        train_data = []
        train_labels = []
        train_identifiers = []
        for data, label, basename in slp_data:
                train_data.append(data)
                train_labels.append(np.concatenate((np.full((data.shape[0],1), label[0]), np.full((data.shape[0],1), label[1])),axis=1)) 
                train_identifiers.append(np.full(data.shape[0], basename)) 
            
                
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
        train_identifiers = np.concatenate(train_identifiers)

        return {'data':train_data, 'label': train_labels, 'identifiers': train_identifiers}
    else:
        dev_data = {}
        for data, label, basename in slp_data:
                if basename not in dev_data:
                    if expand_labels:
                        dev_data[basename] = {'data':[], 'label':[]}
                    else:
                        dev_data[basename] = {'data':[], 'label':label}
                        
                dev_data[basename]['data'].append(data)
                if expand_labels:
                    dev_data[basename]['label'].append(np.concatenate((np.full((data.shape[0],1), label[0]), np.full((data.shape[0],1), label[1])),axis=1)) 
            
        ## We concatenate all the frames belonging to the same filename
        for basename in dev_data:
                dev_data[basename]['data'] = np.concatenate(dev_data[basename]['data'])
                if expand_labels:
                    dev_data[basename]['label'] = np.concatenate(dev_data[basename]['label'])
                    
        return dev_data
    

def save_model(model, model_id, path):
    """
    Save the model to a file.
    """
  
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    model_name = f'{model_id}_{now}'
    os.mkdir(f'{path}/{model_name}/')

    filename = f'{path}/{model_name}/model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    
    print(f"Model saved to {filename}")
    return model_name

def load_model(filename):
    """
    Load the model from a file.
    """
    if not os.path.isfile(filename):
        raise RuntimeError(f"Model file {filename} does not exist.")
    
    model = pickle.load(open(filename, 'rb'))
    return model


def gmm_predict(models, dev_data, CLASSES, label_pos=0):
    """
    Predict the language of the input data using the GMM model.
    """

    results_dev = {}
    results_dev['ref'] =  list()
    results_dev['hyp'] =  list()
    results_dev['llhs'] = np.empty((len(dev_data), len(CLASSES)), dtype=np.float64)
    results_dev['fileids'] = list()

    for i, fileid in tqdm(enumerate(sorted(dev_data)), total=len(dev_data)):
        data = dev_data[fileid]['data']  # the features
        
        results_dev['fileids'].append(fileid)     #fileid

        # store the reference. Notice that we only have this for the dev set, not for the eval
        results_dev['ref'].append(dev_data[fileid]['label'][label_pos]) #reference

        # obtain the log-likelihood score for each model and store
        results_dev['llhs'][i,:] = np.array([models[lang].score(data) for lang in CLASSES])

        
        # Obtain the maximum likelihood nativelanguge estimation
        ix = np.argmax(results_dev['llhs'][i,:])
        results_dev['hyp'].append(CLASSES[ix])
        
    return results_dev
    
def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.matshow(cm, cmap=cmap, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center')

    # Set axis labels and ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)


    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title(title, fontsize=12)
    plt.show()
    
    

def create_submission_file(resdir, filename):


    results_dev = pickle.load(open(f'{resdir}/dev.pkl', 'rb'))
    results_evl = pickle.load(open(f'{resdir}/evl.pkl', 'rb'))

    with open(filename, 'w', newline='') as file:
        csv_writer = csv.writer(file) # CSV writer
        csv_writer.writerow(('fileId', 'Age')) # Header of the CSV

        # Save dev results
        for i in range(len(results_dev['fileids'])):
            csv_writer.writerow((results_dev['fileids'][i], f'{results_dev["hyp"][i]:.1f}'))
        # Save evl results
        for i in range(len(results_evl['fileids'])):
            csv_writer.writerow((results_evl['fileids'][i], f'{results_evl["hyp"][i]:.1f}'))


def predict_nn(model, dataset, target_pos=1):
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False
    )
    predictions = []
    references = []
    fileids = []
    
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        for batch_X, batch_y, batch_files in test_loader:
            outputs = model(batch_X.squeeze(dim=1))
            predictions.append(outputs.item())
            
            # 1. Extract the raw target safely
            raw_target = batch_y[target_pos][0] if isinstance(batch_y, (list, tuple)) else batch_y[0]
            
            # 2. Check if it's a tensor (has .item()) or a native int/str
            val = raw_target.item() if hasattr(raw_target, 'item') else raw_target
            
            # 3. Handle the "?" string for missing ages in the 'evl' set
            if val == "?":
                val = np.nan
            else:
                val = float(val)
                
            references.append(val)
            fileids.append(batch_files[0])
    
    return np.array(predictions), np.array(references, dtype=float), np.array(fileids)

def train_nn(model, traindataset, testdataset, target_pos=1, batch_size=16, epochs=200, lr=0.005, momentum=0.9, weight_decay=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    train_loader = torch.utils.data.DataLoader(
            dataset=traindataset,
            batch_size=batch_size,
            shuffle=True
    )
    
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0.0
        valid_batches = 0
        
        for batch_X, batch_y, _ in train_loader:
            outputs = model(batch_X.squeeze(dim=1))
            
            # Extract raw targets (could be a mix of ints and "?" strings)
            raw_targets = batch_y[target_pos] if isinstance(batch_y, (list, tuple)) else batch_y
            
            clean_targets = []
            clean_outputs = []
            
            # Filter out missing "?" targets from the batch
            for i, t in enumerate(raw_targets):
                val = t.item() if hasattr(t, 'item') else t
                if val != "?":
                    clean_targets.append(float(val))
                    clean_outputs.append(outputs[i])
            
            # Skip the backward pass if all labels in this batch were missing
            if not clean_targets:
                continue
                
            # Convert cleaned lists to tensors for Loss calculation
            targets_tensor = torch.tensor(clean_targets, dtype=torch.float32).view(-1, 1)
            outputs_tensor = torch.stack(clean_outputs).view(-1, 1)
            
            loss = criterion(outputs_tensor, targets_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
            print(f'Epoch {epoch+1}/{epochs}, Train MSE Loss: {avg_loss:.4f}')
            
            if testdataset is not None:
                hyp, ref, _ = predict_nn(model, testdataset, target_pos=target_pos)
                
                # Calculate MAE, ignoring the NaNs we created from "?" labels
                valid_mask = ~np.isnan(ref)
                if valid_mask.sum() > 0:
                    mae = np.mean(np.abs(hyp[valid_mask] - ref[valid_mask]))
                    print(f'Dev Mean Absolute Error (MAE): {mae:.4f}')
                else:
                    print('Dev Mean Absolute Error (MAE): N/A (No known dev labels)')


def extract_and_verify_data_for_ssl(datadir, partition_name):
    """Locate the audio directory for the SSL notebook and normalize info.csv.

    The teacher's SSL notebook expects a helper that:
    - validates the partition's info.csv
    - coerces the age column to numeric values where possible
    - discovers whether audio lives under wav/, audio/, or directly in the partition
    """

    print(f"\n--- Preparing data for partition: {partition_name} ---")
    partition_dir = os.path.join(datadir, partition_name)
    info_csv_path = os.path.join(partition_dir, "info.csv")

    if not os.path.exists(info_csv_path):
        raise FileNotFoundError(
            f"Expected info.csv not found for partition '{partition_name}' at "
            f"'{info_csv_path}'. Please ensure data is extracted."
        )

    df_info = pd.read_csv(info_csv_path)
    if df_info.empty:
        raise ValueError(f"info.csv for partition '{partition_name}' is empty.")
    if "age" not in df_info.columns:
        raise ValueError(
            f"info.csv for partition '{partition_name}' does not contain 'age' column."
        )
    if "wav" not in df_info.columns:
        raise ValueError(
            f"info.csv for partition '{partition_name}' does not contain 'wav' column."
        )

    df_info["age"] = pd.to_numeric(df_info["age"], errors="coerce")
    df_info.to_csv(info_csv_path, index=False)

    if df_info.empty:
        raise ValueError(
            f"info.csv for partition '{partition_name}' is unexpectedly empty."
        )

    sample_wav_filename = df_info.iloc[0]["wav"]
    audio_base_path = None
    potential_audio_dirs = [
        os.path.join(partition_dir, "audio"),
        os.path.join(partition_dir, "wav"),
        partition_dir,
    ]

    for candidate in potential_audio_dirs:
        if os.path.exists(os.path.join(candidate, sample_wav_filename)):
            audio_base_path = candidate
            break

    if audio_base_path is None:
        print(
            f"Warning: Common audio subdirectories not found for '{partition_name}'. "
            "Attempting deeper search..."
        )
        for root, _, files in os.walk(partition_dir):
            if sample_wav_filename in files:
                audio_base_path = root
                break

    if audio_base_path is None:
        raise FileNotFoundError(
            f"Could not locate audio files for partition '{partition_name}' based on "
            f"'{sample_wav_filename}' from info.csv. No audio directory found within "
            f"{partition_dir}."
        )

    print(f"Audio files for '{partition_name}' are located in: {audio_base_path}")
    return audio_base_path
