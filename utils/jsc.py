#  Copyright (C) 2021 Xilinx, Inc
#  Copyright (C) 2020 FastML
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import h5py
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity
import os


# Based off example from: https://github.com/hls-fpga-machine-learning/pytorch-training/blob/master/train/Data_loader.py
# Creates a PyTorch Dataset from the h5 file input.
# Returns labels as a one-hot encoded vector.
# Input / output labels are contained in self.feature_labels / self.output_labels respectively
class JetSubstructureDataset(Dataset):

    file_list = [
        ('https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v/download',
         '3b91f16a1949cb6cf855442867cc26a1'),
    ]

    inputs = ['j_zlogz', 'j_c1_b0_mmdt', 'j_c1_b1_mmdt', 'j_c1_b2_mmdt', 'j_c2_b1_mmdt', 'j_c2_b2_mmdt', 'j_d2_b1_mmdt',
              'j_d2_b2_mmdt', 'j_d2_a1_b1_mmdt', 'j_d2_a1_b2_mmdt', 'j_m2_b1_mmdt', 'j_m2_b2_mmdt', 'j_n2_b1_mmdt',
              'j_n2_b2_mmdt', 'j_mass_mmdt', 'j_multiplicity']

    labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']

    def __init__(self, root, split="train", thresholds=10, download=False, normalize_inputs=True, apply_pca=False):
        super().__init__()

        self.root = root
        self.split = split

        if download and not self._check_integrity():
            self.downloads()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        with h5py.File(os.path.join(root, "download"), 'r') as h5py_file:
            tree_array = h5py_file["t_allpar_new"][()]

        self.feature_labels = self.inputs
        self.output_labels = self.labels

        # Filter input file and convert inputs / outputs to numpy array
        dataset_df = pd.DataFrame(
            tree_array,
            columns=list(set(self.feature_labels + self.output_labels)))
        dataset_df = dataset_df.drop_duplicates()
        features_df = dataset_df[self.feature_labels]
        outputs_df = dataset_df[self.output_labels]
        X = features_df.values
        y = outputs_df.values
        if "j_index" in self.feature_labels:
            X = X[:, :-1]  # drop the j_index feature
        if "j_index" in self.output_labels:
            # drop the j_index label
            y = y[:, :-1]
            self.output_labels = self.output_labels[:-1]
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )  # Using the same dataset split as: https://github.com/hls-fpga-machine-learning/pytorch-training/blob/master/train/Data_loader.py
        if normalize_inputs:
            # scaler = preprocessing.StandardScaler().fit(X_train_val)
            scaler = preprocessing.MinMaxScaler().fit(X_train_val)
            X_train_val = scaler.transform(X_train_val)
            X_test = scaler.transform(X_test)

        if apply_pca:
            # Apply dimenionality reduction to the inputs
            with torch.no_grad():
                dim = self.config["PcaDimensions"]
                X_train_val_fp64 = torch.from_numpy(X_train_val).double()
                X_test_fp64 = torch.from_numpy(X_test).double()
                U, S, V = torch.svd(X_train_val_fp64)
                X_train_val_pca_fp64 = torch.mm(X_train_val_fp64, V[:, 0:dim])
                X_test_pca_fp64 = torch.mm(X_test_fp64, V[:, 0:dim])
                variance_retained = 100 * (S[0:dim].sum() / S.sum())
                print(f"Dimensions used for PCA: {dim}")
                print(f"Variance retained (%): {variance_retained}")
                X_train_val = X_train_val_pca_fp64.float().numpy()
                X_test = X_test_pca_fp64.float().numpy()

        X_train_val = torch.Tensor(np.array([(X_train_val > (i + 1) / thresholds) for i in range(thresholds-1)]))
        X_train_val = np.reshape(np.transpose(X_train_val, (1,0,2)), (X_train_val.shape[1],-1))
        X_test = torch.Tensor(np.array([(X_test > (i + 1) / thresholds) for i in range(thresholds - 1)]))
        X_test = np.reshape(np.transpose(X_test, (1,0,2)), (X_test.shape[1],-1))

        if self.split == "train":
            self.X = X_train_val  # torch.from_numpy(X_train_val)[:, None, :]
            # self.y = torch.from_numpy(y_train_val).float()
            self.y = torch.argmax(torch.from_numpy(y_train_val), dim=1)
        elif self.split == "test":
            self.X = X_test  # torch.from_numpy(X_test)[:, None, :]
            # self.y = torch.from_numpy(y_test).float()
            self.y = torch.argmax(torch.from_numpy(y_test), dim=1)

    def _check_integrity(self):
        for file in self.file_list:
            md5 = file[1]
            fpath = os.path.join(self.root, file[0].split('/')[-1])
            if not check_integrity(fpath, md5):
                return False
        return True

    def downloads(self):
        for file in self.file_list:
            md5 = file[1]
            download_url(file[0], self.root, file[0].split('/')[-1], md5)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

if __name__ == '__main__':
    thresholds =1000
    train_set = JetSubstructureDataset("./data-jsc", thresholds=thresholds, split="train", download=True)
    test_set = JetSubstructureDataset("./data-jsc", thresholds=thresholds, split="test", download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    input_dim_dataset = 16*(thresholds-1)
    num_classes = 5

    for batch in train_loader:
        x, y = batch
        print(1)
