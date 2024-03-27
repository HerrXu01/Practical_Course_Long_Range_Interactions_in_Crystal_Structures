from jarvis.db.figshare import data as jdata
from torch_geometric.data import Data
import torch
import periodictable
import os
import lmdb
from tqdm import tqdm
import pickle
import random
import numpy as np


def atom2number(elements):
    atom_numbers = []
    for ele in elements:
        atom_numbers.append(periodictable.elements.symbol(ele).number)
    
    return torch.tensor(atom_numbers)

def get_id_train_val_test(
        total_size=1000,
        split_seed=123,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        n_train=None,
        n_test=None,
        n_val=None,
        keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
            train_ratio is None
            and val_ratio is not None
            and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1

    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)

    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test): -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test

def split_dataset(data: list, id_train: list, id_val: list, id_test: list):
    dataset_train = [data[x] for x in id_train]
    dataset_val = [data[x] for x in id_val]
    dataset_test = [data[x] for x in id_test]

    target = []
    for d in dataset_train:
        target.append(d["formation_energy_peratom"])
    target_mean = torch.tensor(target).mean()
    target_std = torch.tensor(target).std()

    return dataset_train, dataset_val, dataset_test, target_mean, target_std

def write_lmdb(dataset: list, dirname: str, filename: str):
    alldata = []
    for i in dataset:
        pos = torch.tensor(i["atoms"]["coords"])
        cell = torch.tensor(i["atoms"]["lattice_mat"]).unsqueeze(0)
        atomic_numbers = atom2number(i["atoms"]["elements"])
        natoms = atomic_numbers.shape[0]
        tags = None
        y = i["formation_energy_peratom"]
        force = None
        fixed = None
        sid = None
        fid = None
        id = None
        
        args = {
            'pos': pos,
            'cell': cell,
            'atomic_numbers': atomic_numbers,
            'natoms': natoms,
            'tags': tags,
            'y': y,
            'force': force,
            'fixed': fixed,
            'sid': sid,
            'fid': fid,
            'id': id
        }

        alldata.append(Data(**args))
    
    os.makedirs(dirname, exist_ok=True)
    path = dirname + '/' + filename
    db = lmdb.open(
        path,
        map_size=1099511627776,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for idx, data in tqdm(enumerate(alldata), total=len(alldata)):
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    txn = db.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(len(alldata), protocol=-1))
    txn.commit()

    db.sync()
    db.close()

d = jdata("dft_3d")
# print(d[0]["atoms"]["coords"])
# print(torch.tensor(d[0]["atoms"]["lattice_mat"]).unsqueeze(0).shape)
id_train, id_val, id_test = get_id_train_val_test(total_size=len(d), split_seed=123, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, keep_data_order=False)
dataset_train, dataset_val, dataset_test, target_mean, target_std = split_dataset(d, id_train, id_val, id_test)
write_lmdb(dataset=dataset_train, dirname="jarvis_lmdb/train", filename="jarvis_train.lmdb")
write_lmdb(dataset=dataset_val, dirname="jarvis_lmdb/val", filename="jarvis_val.lmdb")
write_lmdb(dataset=dataset_test, dirname="jarvis_lmdb/test", filename="jarvis_test.lmdb")
print("The following mean and std are for the dataset in config.yml.")
print("target mean: ", target_mean)
print("target std: ", target_std)


'''
alldata = []
for i in d:
    pos = torch.tensor(i["atoms"]["coords"])
    cell = torch.tensor(i["atoms"]["lattice_mat"]).unsqueeze(0)
    atomic_numbers = atom2number(i["atoms"]["elements"])
    natoms = atomic_numbers.shape[0]
    tags = None
    y = i["formation_energy_peratom"]
    force = None
    fixed = None
    sid = None
    fid = None
    id = None
    
    args = {
        'pos': pos,
        'cell': cell,
        'atomic_numbers': atomic_numbers,
        'natoms': natoms,
        'tags': tags,
        'y': y,
        'force': force,
        'fixed': fixed,
        'sid': sid,
        'fid': fid,
        'id': id
    }

    alldata.append(Data(**args))

os.makedirs("jarvis_lmdb", exist_ok=True)
db = lmdb.open(
    "jarvis_lmdb/jarvis.lmdb",
    map_size=1099511627776,
    subdir=False,
    meminit=False,
    map_async=True,
)

for idx, data in tqdm(enumerate(alldata), total=len(alldata)):
    txn = db.begin(write=True)
    txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()

txn = db.begin(write=True)
txn.put(f"length".encode("ascii"), pickle.dumps(len(alldata), protocol=-1))
txn.commit()

db.sync()
db.close()
'''
