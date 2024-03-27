from models.config import TrainingConfigPotnet, TrainingConfigMatformer
from models.potnet import PotNet
from models.pyg_att import Matformer

import torch
import ignite
import os
import time
import numpy as np
import tqdm
import pickle as pk
import pprint
import json

from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.contrib.handlers import TensorboardLogger
from ignite.handlers.stores import EpochOutputStore
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from torch import nn
#from jarvis.db.jsonutils import loadjson
from jarvis.db.figshare import data as jdata
from typing import Any, Dict, Union
from jarvis.db.jsonutils import dumpjson
from sklearn.metrics import mean_absolute_error

from data import get_train_val_loaders

from functools import partial

import datetime
import wandb
wandb.login()

device = torch.device("cuda")

criteria = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
}

def set_scheduler(config, optimizer, train_loader):
    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        print(len(train_loader), config.epochs)
        pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.5
        )
    return scheduler

def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]

def prepare_batch_potnet(
        batch, device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    batch = (
        batch.to(device, non_blocking=non_blocking),
        batch.y.to(device, non_blocking=non_blocking),
    )

    return batch

def setup_optimizer(params, config):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer

def get_data(config):
    #TODO:Custom datasets/dataloaders
    if(config.model.name == "potnet"):
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
            mean,
            std
        ) = get_train_val_loaders(
            model_type=config.model.name,
            dataset=config.dataset,
            root=config.output_dir,
            cachedir=config.cache_dir,
            processdir=config.process_dir,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            normalize=config.normalize,
            euclidean=config.euclidean,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            infinite_funcs=config.infinite_funcs,
            infinite_params=config.infinite_params,
            R=config.R,
            keep_data_order=config.keep_data_order,
        )
    elif(config.model.name == "matformer"):
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
            mean,
            std,
        ) = get_train_val_loaders(
            model_type=config.model.name,
            dataset=config.dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=config.line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            matrix_input=config.matrix_input,
            pyg_input=config.pyg_input,
            use_lattice=config.use_lattice,
            use_angle=config.use_angle,
            use_save=config.use_save,
            mp_id_list=None,
        )
    return train_loader, val_loader, test_loader, prepare_batch, mean, std

def get_model(config, checkpoint):
    if(config.model.name == "potnet"):
        _model = {
            "potnet": PotNet,
        }
        config.model.euclidean = config.euclidean
    if(config.model.name == "matformer"):
        _model = {
            "matformer": Matformer,
        }

    net = _model.get(config.model.name)(config.model)
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint)["model"])
    net.to(device)
    return net

def train(config, model, optimizer, metrics, criterion, scheduler, std, mean, train_loader, val_loader, test_loader, prepare_batch, testing):
    if config.use_wandb:
        wandb.init(project = "Potnet_small", entity  = 'labcourse_wise23_project4', config = config.dict(), name = "EwaldAP_0.5kcutoff", save_code=True)
        model.run_id = wandb.run.id

    deterministic = True
    checkpoint_dir = os.path.join(config.output_dir, config.checkpoint_dir)
    if(prepare_batch is None):
        prepare_batch = prepare_batch_potnet
    else:
        prepare_batch = partial(prepare_batch, device=device)

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
    )

    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    train_evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=5,
            score_name="neg_mae",
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    if config.store_outputs:
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        train_evaluator.run(train_loader)
        evaluator.run(val_loader)

        tmetrics = train_evaluator.state.metrics
        vmetrics = evaluator.state.metrics
        for metric in metrics.keys():
            tm = tmetrics[metric]
            vm = vmetrics[metric]
            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()
                vm = vm.cpu().numpy().tolist()

            history["train"][metric].append(tm)
            history["validation"][metric].append(vm)

        if config.use_wandb:
            wandb.log({
                "epoch": engine.state.epoch,
                "train_loss": tmetrics["loss"],
                "train_mae": tmetrics["mae"],
                "val_loss": vmetrics["loss"],
                "val_mae": vmetrics["mae"]
            })

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, config.model.name + "_" + config.target + "_history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir,
                                      config.model.name + "_" + config.target + "_history_train.json"),
                data=history["train"],
            )
        if config.progress:
            pbar = ProgressBar()
            pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
            pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")

    if config.n_early_stopping is not None:
        def default_score_fn(engine):
            score = engine.state.metrics["mae"]
            return score

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=default_score_fn,
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)

    if config.log_tensorboard:

        tb_logger = TensorboardLogger(
            log_dir=os.path.join(config.output_dir, "tb_logs", "test")
        )
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "mae", "neg_mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    # train the model!
    print("Training!")
    if not testing:
        trainer.run(train_loader, max_epochs=config.epochs)
    if config.log_tensorboard:
        test_loss = evaluator.state.metrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()

    print("Testing!")
    model.eval()
    t1 = time.time()
    if config.write_predictions:
        f = open(
            os.path.join(config.output_dir,
                         config.model.name + "_" + config.target + "_prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
    targets = []
    predictions = []

    with torch.no_grad():
        ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
        if(config.model.name == "potnet"):
            for dat in test_loader:
                data = dat
                target = dat.label
                out_data = model(data.to(device))
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()
                if isinstance(out_data, list) and len(out_data) == 1:
                    out_data = out_data[0]
                if len(target) == 1:
                    target = target[0]

                out_data = out_data * std + mean
                if config.write_predictions:
                    f.write("%s, %6f, %6f\n" % (id, target, out_data))

                targets.append(target)
                predictions.append(out_data)
        elif(config.model.name == "matformer"):
            for dat in test_loader:
                g, lg, _, target = dat
                out_data = model([g.to(device), lg.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()
                if len(target) == 1:
                    target = target[0]
                targets.append(target)
                predictions.append(out_data)
    if config.write_predictions:
        f.close()
    t2 = time.time()
    print("Test time(s):", t2 - t1)


    print(
        "Test MAE:",
        mean_absolute_error(np.array(targets), np.array(predictions)),
    )
    if config.use_wandb:
        wandb.finish()
    return mean_absolute_error(np.array(targets), np.array(predictions))
    

def train_model(config_dict: Dict, data_root:str = None, checkpoint: str = None, testing: bool = False, file_format: str = 'poscar'):
    if config_dict["dataset"] == "megnet":
        config_dict["id_tag"] = "id"
        if config_dict["target"] == "e_form" or config_dict["target"] == "gap pbe":
            config_dict["n_train"] = 4000
            config_dict["n_val"] = 500
            config_dict["n_test"] = 500

    print("Start")
    if(config_dict["model"]["name"] == "potnet"):
        config = TrainingConfigPotnet(**config_dict)
    elif(config_dict["model"]["name"] == "matformer"):
        print("Config dict: ", config_dict)
        config = TrainingConfigMatformer(**config_dict)
        print("Config: ", config)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    print('output_dir train', config.output_dir)


    #TODO: Config random seed
    print("Geting data")
    train_loader, val_loader, test_loader, prepare_batch, mean, std = get_data(config)
    if std is None:
        std = 1.0
    print("Geting model")
    model = get_model(config, checkpoint)

    print("Setting optimizer")
    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(model)

    # Setup optimizer and scheduler to config value
    optimizer = setup_optimizer(params, config)
    scheduler = set_scheduler(config, optimizer, train_loader)

    # Select criteria from config file
    criterion = criteria[config.criterion]

    # Metrics: loss(criteria from config), Mean Absolute error
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError() * std, "neg_mae": -1.0 * MeanAbsoluteError() * std}

    # Training process
    return train(config, 
                 model, 
                 optimizer, 
                 metrics, 
                 criterion, 
                 scheduler, 
                 std, 
                 mean, 
                 train_loader, 
                 val_loader, 
                 test_loader, 
                 prepare_batch, 
                 testing)

