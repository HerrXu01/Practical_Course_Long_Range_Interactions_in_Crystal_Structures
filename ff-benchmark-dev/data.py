import itertools
import random
import sys
import time
from pathlib import Path
from typing import Optional

import os
import torch
import numpy as np
import pandas as pd
from jarvis.db.figshare import data as jdata
from jarvis.core.specie import chem_data, get_node_attributes

from torch.utils.data import DataLoader as DataLoaderGeneral
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson

import pickle as pk

from sklearn.preprocessing import StandardScaler

from pandarallel import pandarallel

from graphs import PygStructureDataset, StructureDataset, load_infinite_graphs, load_radius_graphs, load_pyg_graphs, mean_absolute_deviation

pandarallel.initialize(progress_bar=True)

tqdm.pandas()

torch.set_printoptions(precision=10)


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


# Get Dataset for PotNet
def get_torch_dataset(
        dataset=None,
        root="",
        cachedir="",
        processdir="",
        name="",
        id_tag="jid",
        target="",
        atom_features="",
        normalize=False,
        euclidean=False,
        cutoff=4.0,
        max_neighbors=16,
        infinite_funcs=[],
        infinite_params=[],
        R=5,
        mean=0.0,
        std=1.0,
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    vals = df[target].values
    print("data range", np.max(vals), np.min(vals))
    cache = os.path.join(root, cachedir)
    if not os.path.exists(cache):
        os.makedirs(cache)
    if euclidean:
        load_radius_graphs(
            df,
            radius=cutoff,
            max_neighbors=max_neighbors,
            name=name + "-" + str(cutoff),
            target=target,
            cachedir=Path(cache),
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{cutoff}-{target}-radius.bin"),
            processdir,
            target=target,
            name=f"{name}-{cutoff}-{target}-radius",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    else:
        load_infinite_graphs(
            df,
            name=name,
            target=target,
            cachedir=Path(cache),
            infinite_funcs=infinite_funcs,
            infinite_params=infinite_params,
            R=R,
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{target}-infinite.bin"),
            processdir,
            target=target,
            name=f"{name}-{target}-infinite",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    return data

# Get dataset for Matformer
def get_pyg_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
    use_lattice=False,
    use_angle=False,
    data_from='Jarvis',
    use_save=False,
    mean_train=None,
    std_train=None,
    now=False, # for test
):
    """Get pyg Dataset."""
    df = pd.DataFrame(dataset)
    # print("df", df)
    # neighbor_strategy = "pairwise-k-nearest"
    
    vals = df[target].values
    if target == "shear modulus" or target == "bulk modulus":
        val_list = [vals[i].item() for i in range(len(vals))]
        vals = val_list
    output_dir = "./saved_data/" + tmp_name + "test_graph_angle.pkl" # for fast test use
    print("data range", np.max(vals), np.min(vals))
    print(output_dir)
    if now:
        if not os.path.exists(output_dir):
            graphs = load_pyg_graphs(
                df,
                name=name,
                neighbor_strategy=neighbor_strategy,
                use_canonize=use_canonize,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            with open(output_dir, 'wb') as pf:
                pk.dump(graphs, pf)
            print('save graphs to ', output_dir)
        else:
            print('loading graphs from ', output_dir)
            with open(output_dir, 'rb') as pf:
                graphs = pk.load(pf)
    else:
        print('graphs not saved')
        graphs = load_pyg_graphs(
            df,
            name=name,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            use_lattice=use_lattice,
            use_angle=use_angle,
        )
    
    if mean_train == None:
        mean_train = np.mean(vals)
        std_train = np.std(vals)
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    else:
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    return data, mean_train, std_train


# Combined loaders, use model_type = 'model_name' to initialize desired model
def get_train_val_loaders(
    model_type,
    dataset: str = "dft_3d",
    root: str = "",
    cachedir: str = "",
    processdir: str = "",
    dataset_array=None,
    target: str = "formation_energy_peratom",
    atom_features: str = "cgcnn",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size: int = 64,
    split_seed: int = 123,
    keep_data_order=False,
    workers: int = 4,
    pin_memory: bool = True,
    id_tag: str = "jid",
    standardize: bool = False,  # Specific to Matformer
    line_graph: bool = True,  # Specific to Matformer
    save_dataloader: bool = False,  # Specific to Matformer
    filename: str = "sample",  # Specific to Matformer
    use_canonize: bool = False,  # Specific to Matformer
    cutoff: float = 8.0,  # Specific to Matformer
    max_neighbors: int = 12,  # Specific to Matformer
    neighbor_strategy: str = "k-nearest", # Specific to Matformer
    classification_threshold: Optional[float] = None,  # Specific to Matformer
    target_multiplication_factor: Optional[float] = None,  # Specific to Matformer
    standard_scalar_and_pca=False,  # Specific to Matformer
    output_features=1,  # Specific to Matformer
    output_dir=None,  # Specific to Matformer
    matrix_input=False,  # Specific to Matformer
    pyg_input=False,  # Specific to Matformer
    use_lattice=False,  # Specific to Matformer
    use_angle=False,  # Specific to Matformer
    use_save=True,  # Specific to Matformer
    mp_id_list=None,  # Specific to Matformer
    normalize=False,  # Specific to Potnet
    euclidean=False,  # Specific to Potnet
    infinite_funcs=[],  # Specific to Potnet
    infinite_params=[],  # Specific to Potnet
    R=5,  # Specific to Potnet
):
    # Matformer
    if model_type == "matformer":
        """Help function to set up JARVIS train and val dataloaders."""
        # data loading
        mean_train=None
        std_train=None
        assert (matrix_input and pyg_input) == False
        
        train_sample = filename + "_train.data"
        val_sample = filename + "_val.data"
        test_sample = filename + "_test.data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if (
            os.path.exists(train_sample)
            and os.path.exists(val_sample)
            and os.path.exists(test_sample)
            and save_dataloader
        ):
            print("Loading from saved file...")
            print("Make sure all the DataLoader params are same.")
            print("This module is made for debugging only.")
            train_loader = torch.load(train_sample)
            val_loader = torch.load(val_sample)
            test_loader = torch.load(test_sample)
            if train_loader.pin_memory != pin_memory:
                train_loader.pin_memory = pin_memory
            if test_loader.pin_memory != pin_memory:
                test_loader.pin_memory = pin_memory
            if val_loader.pin_memory != pin_memory:
                val_loader.pin_memory = pin_memory
            if train_loader.num_workers != workers:
                train_loader.num_workers = workers
            if test_loader.num_workers != workers:
                test_loader.num_workers = workers
            if val_loader.num_workers != workers:
                val_loader.num_workers = workers
            print("train", len(train_loader.dataset))
            print("val", len(val_loader.dataset))
            print("test", len(test_loader.dataset))
            return (
                train_loader,
                val_loader,
                test_loader,
                train_loader.dataset.prepare_batch,
            )
        else:
            if not dataset_array:
                d = jdata(dataset)
            else:
                d = dataset_array
                # for ii, i in enumerate(pc_y):
                #    d[ii][target] = pc_y[ii].tolist()

            dat = []
            if classification_threshold is not None:
                print(
                    "Using ",
                    classification_threshold,
                    " for classifying ",
                    target,
                    " data.",
                )
                print("Converting target data into 1 and 0.")
            all_targets = []

            # TODO:make an all key in qm9_dgl
            if dataset == "qm9_dgl" and target == "all":
                print("Making all qm9_dgl")
                tmp = []
                for ii in d:
                    ii["all"] = [
                        ii["mu"],
                        ii["alpha"],
                        ii["homo"],
                        ii["lumo"],
                        ii["gap"],
                        ii["r2"],
                        ii["zpve"],
                        ii["U0"],
                        ii["U"],
                        ii["H"],
                        ii["G"],
                        ii["Cv"],
                    ]
                    tmp.append(ii)
                print("Made all qm9_dgl")
                d = tmp
            for i in d:
                if isinstance(i[target], list):  # multioutput target
                    all_targets.append(torch.tensor(i[target]))
                    dat.append(i)

                elif (
                    i[target] is not None
                    and i[target] != "na"
                    and not math.isnan(i[target])
                ):
                    if target_multiplication_factor is not None:
                        i[target] = i[target] * target_multiplication_factor
                    if classification_threshold is not None:
                        if i[target] <= classification_threshold:
                            i[target] = 0
                        elif i[target] > classification_threshold:
                            i[target] = 1
                        else:
                            raise ValueError(
                                "Check classification data type.",
                                i[target],
                                type(i[target]),
                            )
                    dat.append(i)
                    all_targets.append(i[target])

        
        if mp_id_list is not None:
            if mp_id_list == 'bulk':
                print('using mp bulk dataset')
                with open('./data/bulk_megnet_train.pkl', 'rb') as f:
                    dataset_train = pk.load(f)
                with open('./data/bulk_megnet_val.pkl', 'rb') as f:
                    dataset_val = pk.load(f)
                with open('./data/bulk_megnet_test.pkl', 'rb') as f:
                    dataset_test = pk.load(f)
            
            if mp_id_list == 'shear':
                print('using mp shear dataset')
                with open('./data/shear_megnet_train.pkl', 'rb') as f:
                    dataset_train = pk.load(f)
                with open('./data/shear_megnet_val.pkl', 'rb') as f:
                    dataset_val = pk.load(f)
                with open('./data/shear_megnet_test.pkl', 'rb') as f:
                    dataset_test = pk.load(f)

        else:
            id_train, id_val, id_test = get_id_train_val_test(
                total_size=len(dat),
                split_seed=split_seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                n_train=n_train,
                n_test=n_test,
                n_val=n_val,
                keep_data_order=keep_data_order,
            )
            ids_train_val_test = {}
            ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
            ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
            ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
            dumpjson(
                data=ids_train_val_test,
                filename=os.path.join(output_dir, "ids_train_val_test.json"),
            )
            dataset_train = [dat[x] for x in id_train]
            dataset_val = [dat[x] for x in id_val]
            dataset_test = [dat[x] for x in id_test]

            # Filter dataset for Uranium
            dataset_train = [element for element in dataset_train if not any(x in element['atoms']['elements'] for x in ['U', 'Ac', 'Th', 'Np', 'Pa', 'Pu'])]
            dataset_val = [element for element in dataset_val if not any(x in element['atoms']['elements'] for x in ['U', 'Ac', 'Th', 'Np', 'Pa', 'Pu'])]
            dataset_test = [element for element in dataset_test if not any(x in element['atoms']['elements'] for x in ['U', 'Ac', 'Th', 'Np', 'Pa', 'Pu'])]

        if standard_scalar_and_pca:
            y_data = [i[target] for i in dataset_train]
            # pipe = Pipeline([('scale', StandardScaler())])
            if not isinstance(y_data[0], list):
                print("Running StandardScalar")
                y_data = np.array(y_data).reshape(-1, 1)
            sc = StandardScaler()

            sc.fit(y_data)
            print("Mean", sc.mean_)
            print("Variance", sc.var_)
            try:
                print("New max", max(y_data))
                print("New min", min(y_data))
            except Exception as exp:
                print(exp)
                pass
            
            pk.dump(sc, open(os.path.join(output_dir, "sc.pkl"), "wb"))

        if classification_threshold is None:
            try:
                from sklearn.metrics import mean_absolute_error

                print("MAX val:", max(all_targets))
                print("MIN val:", min(all_targets))
                print("MAD:", mean_absolute_deviation(all_targets))
                try:
                    f = open(os.path.join(output_dir, "mad"), "w")
                    line = "MAX val:" + str(max(all_targets)) + "\n"
                    line += "MIN val:" + str(min(all_targets)) + "\n"
                    line += (
                        "MAD val:"
                        + str(mean_absolute_deviation(all_targets))
                        + "\n"
                    )
                    f.write(line)
                    f.close()
                except Exception as exp:
                    print("Cannot write mad", exp)
                    pass
                # Random model precited value
                x_bar = np.mean(np.array([i[target] for i in dataset_train]))
                baseline_mae = mean_absolute_error(
                    np.array([i[target] for i in dataset_test]),
                    np.array([x_bar for i in dataset_test]),
                )
                print("Baseline MAE:", baseline_mae)
            except Exception as exp:
                print("Data error", exp)
                pass
        
        train_data, mean_train, std_train = get_pyg_dataset(
            dataset=dataset_train,
            id_tag=id_tag,
            atom_features=atom_features,
            target=target,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            name=dataset,
            line_graph=line_graph,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            classification=classification_threshold is not None,
            output_dir=output_dir,
            tmp_name="train_data",
            use_lattice=use_lattice,
            use_angle=use_angle,
            use_save=False,
        )
        val_data,_,_ = get_pyg_dataset(
            dataset=dataset_val,
            id_tag=id_tag,
            atom_features=atom_features,
            target=target,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            name=dataset,
            line_graph=line_graph,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            classification=classification_threshold is not None,
            output_dir=output_dir,
            tmp_name="val_data",
            use_lattice=use_lattice,
            use_angle=use_angle,
            use_save=False,
            mean_train=mean_train,
            std_train=std_train,
        )
        test_data,_,_ = get_pyg_dataset(
            dataset=dataset_test,
            id_tag=id_tag,
            atom_features=atom_features,
            target=target,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            name=dataset,
            line_graph=line_graph,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            classification=classification_threshold is not None,
            output_dir=output_dir,
            tmp_name="test_data",
            use_lattice=use_lattice,
            use_angle=use_angle,
            use_save=False,
            mean_train=mean_train,
            std_train=std_train,
        )
        
        collate_fn = train_data.collate
        if line_graph:
            collate_fn = train_data.collate_line_graph

        # use a regular pytorch dataloader
        train_loader = DataLoaderGeneral(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoaderGeneral(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoaderGeneral(
            test_data,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin_memory,
        )
        if save_dataloader:
            torch.save(train_loader, train_sample)
            torch.save(val_loader, val_sample)
            torch.save(test_loader, test_sample)
        
        print("n_train:", len(train_loader.dataset))
        print("n_val:", len(val_loader.dataset))
        print("n_test:", len(test_loader.dataset))
        return (
            train_loader,
            val_loader,
            test_loader,
            train_loader.dataset.prepare_batch,
            mean_train,
            std_train,
        )
    
    
    # PotNet
    elif model_type == "potnet":
        if not dataset_array:
            d = jdata(dataset)
        else:
            d = dataset_array

        dat = []
        all_targets = []

        for i in d:
            if isinstance(i[target], list):
                all_targets.append(torch.tensor(i[target]))
                dat.append(i)

            elif (
                    i[target] is not None
                    and i[target] != "na"
                    and not math.isnan(i[target])
            ):
                dat.append(i)
                all_targets.append(i[target])

        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dat),
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_train=n_train,
            n_test=n_test,
            n_val=n_val,
            keep_data_order=keep_data_order,
        )
        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(root, "ids_train_val_test.json"),
        )
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]

        # Filter dataset for heavy atoms
        dataset_train = [element for element in dataset_train if not any(x in element['atoms']['elements'] for x in ['U', 'Ac', 'Th', 'Np', 'Pa', 'Pu'])]
        dataset_val = [element for element in dataset_val if not any(x in element['atoms']['elements'] for x in ['U', 'Ac', 'Th', 'Np', 'Pa', 'Pu'])]
        dataset_test = [element for element in dataset_test if not any(x in element['atoms']['elements'] for x in ['U', 'Ac', 'Th', 'Np', 'Pa', 'Pu'])]

        start = time.time()
        train_data = get_torch_dataset(
            dataset=dataset_train,
            root=root,
            cachedir=cachedir,
            processdir=processdir,
            name=dataset + "_train",
            id_tag=id_tag,
            target=target,
            atom_features=atom_features,
            normalize=normalize,
            euclidean=euclidean,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            infinite_funcs=infinite_funcs,
            infinite_params=infinite_params,
            R=R,
        )

        mean = train_data.mean
        std = train_data.std

        val_data = get_torch_dataset(
            dataset=dataset_val,
            root=root,
            cachedir=cachedir,
            processdir=processdir,
            name=dataset + "_val",
            id_tag=id_tag,
            target=target,
            atom_features=atom_features,
            normalize=normalize,
            euclidean=euclidean,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            infinite_funcs=infinite_funcs,
            infinite_params=infinite_params,
            R=R,
            mean=mean,
            std=std
        )

        test_data = get_torch_dataset(
            dataset=dataset_test,
            root=root,
            cachedir=cachedir,
            processdir=processdir,
            name=dataset + "_test",
            id_tag=id_tag,
            target=target,
            atom_features=atom_features,
            normalize=normalize,
            euclidean=euclidean,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            infinite_funcs=infinite_funcs,
            infinite_params=infinite_params,
            R=R,
            mean=mean,
            std=std,
        )

        print("------processing time------: " + str(time.time() - start))

        # use a graph dataloader
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        print("n_train:", len(train_loader.dataset))
        print("n_val:", len(val_loader.dataset))
        print("n_test:", len(test_loader.dataset))
        return (
            train_loader,
            val_loader,
            test_loader,
            None,
            mean,
            std,
        )
