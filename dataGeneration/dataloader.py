import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class TrainLoader(Dataset):

    def __init__(self, x_files, y_files):

        self.x_files = x_files
        self.y_files = y_files

        self.index_map = []

        for file_id, path in enumerate(x_files):

            x = np.load(path, mmap_mode='r')

            num_samples = x.shape[0]

            for local_idx in range(num_samples):
                self.index_map.append((file_id, local_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        file_id, local_idx = self.index_map[idx]

        x = np.load(self.x_files[file_id], mmap_mode='r')
        y = np.load(self.y_files[file_id], mmap_mode='r')
        
        x_sample = x[local_idx]
        y_sample = y[local_idx]
        
        x_sample = torch.from_numpy(x_sample.copy()).float()
        y_sample = torch.from_numpy(y_sample.copy()).long()

        return x_sample, y_sample




trainXFiles=[]
trainYFiles=[]

baseX='/mnt/SSD/processedData/trainXData/'
baseY='/mnt/SSD/processedData/trainYData/'

root = Path('/mnt/SSD/processedData/trainXData/') # All files for training are stored in this folder. For more information, see "processdata.py"

for a in root.rglob('*'):

    trainXFiles.append(baseX + a.stem + '.npy')
    trainYFiles.append(baseY + a.stem + '.npy')

trainDataset = TrainLoader(trainXFiles,trainYFiles)

numWorker = 4

SEED = 3257354076 # This is the seed used to obtain the results published in the paper

print(f"Seed: {SEED}")

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

trainLoader = DataLoader(
    trainDataset,
    batch_size=256,
    shuffle=True,
    num_workers=numWorker,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=False,
    persistent_workers=False

)
