import torch
from torch.optim import SGD
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import numpy as np
from os import makedirs, listdir
from os.path import join, basename, dirname

try:
    from ConfigSpace import Configuration, ConfigurationSpace
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac import MultiFidelityFacade as MFFacade
    from smac.intensifier.hyperband import Hyperband
except ImportError:
    #print("WARNING: SMAC not installed. Please install it to use SMAC optimization.")
    pass

from usam.MLP import MLP
from usam.training.dataset import ContainerDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def regression_metrics(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)

    l1_norm = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    r2 = r2_score(targets, predictions)
    around_01 = np.sum(np.abs(predictions - targets) < 0.1) / len(targets)
    around_005 = np.sum(np.abs(predictions - targets) < 0.05) / len(targets)
    around_001 = np.sum(np.abs(predictions - targets) < 0.01) / len(targets)
    metrics = {
        "num_samples": len(targets),
        "l1_norm": l1_norm,
        "r2": r2,
        "mse": mse,
        "around_01": around_01,
        "around_005": around_005,
        "around_001": around_001,
    }
    return metrics


def train_epoch(data, model, loss_fn, optimizer, epoch, silent=False):
    accumulated_loss = 0
    model.train()
    if not silent:
        pbar = tqdm(data, desc=f"Train epoch {epoch}", mininterval=1.0)
    else:
        pbar = data

    idx = 0
    for d in pbar:
        # Every data instance is an input + label pair
        inputs, target = d
        inputs = inputs.to(DEVICE).float()
        target = target.to(DEVICE).float()
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(), target.squeeze())
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        accumulated_loss += loss.detach().cpu().item() / len(data)
        if not silent and idx % 50 == 0:
            pbar.set_postfix({"Loss": accumulated_loss})
        idx += 1
    return accumulated_loss


def validate(data, model, epoch=None, loss_fn=None, silent=False):
    accumulated_loss = 0

    targets = []
    predictions = []
    model.eval()
    if not silent and epoch is not None:
        pbar = tqdm(data, desc=f"Eval epoch {epoch}")
    else:
        pbar = data
    for d in pbar:
        # Every data instance is an input + label pair
        inputs, target = d
        inputs = inputs.to(DEVICE).float()
        target = target.to(DEVICE).float()
        # Make predictions for this batch
        outputs = model(inputs)
        targets.append(target.squeeze().cpu().numpy())
        predictions.append(outputs.squeeze().cpu().detach().numpy())
        # Compute the loss and its gradients
        if loss_fn is not None:
            loss = loss_fn(outputs.squeeze(), target.squeeze())
            accumulated_loss += loss.detach().cpu().item() / len(data)
        if not silent and epoch is not None:
            pbar.set_postfix({"Val Loss": accumulated_loss})

    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)

    return targets, predictions, accumulated_loss


def preload_data(
        train_dataset_path, 
        val_dataset_path, 
        metric, 
        filter_model=None, 
        no_augmented=False,
        full_train=False,
):

    if isinstance(train_dataset_path, str):
        if basename(train_dataset_path).startswith("sav_ALL_"):
            sub_sets = list()
            for file in sorted(listdir(dirname(train_dataset_path))):
                if file.startswith("sav_") and file.endswith(basename(train_dataset_path)[8:]):
                    sub_sets.append(file)
            print("Subsets: ", sub_sets)
            if full_train:
                train_datasets = list()
                for sub_set in sub_sets:
                    train_datasets.append(ContainerDataset(join(dirname(train_dataset_path), sub_set), metric))
            else:
                # Only use the first 80% of the data
                train_datasets = list()
                for sub_set in sub_sets[0:int(len(sub_sets)*0.8)]:
                    train_datasets.append(ContainerDataset(join(dirname(train_dataset_path), sub_set), metric))
            # Only use the last 20% of the data
            val_datasets = list()
            for sub_set in sub_sets[int(len(sub_sets)*0.8):]:
                val_datasets.append(ContainerDataset(join(dirname(train_dataset_path), sub_set), metric))

            # Concatenate the torch datasets
            train_data = torch.utils.data.ConcatDataset(train_datasets)
            val_data = torch.utils.data.ConcatDataset(val_datasets)

        else:
            if train_dataset_path == val_dataset_path:
                print("Val Data equals Train Data --> Split train data")
                if full_train:
                    train_data = ContainerDataset(train_dataset_path, metric, )
                else:
                    train_data = ContainerDataset(train_dataset_path, metric,
                                                  "train")
                val_data = ContainerDataset(val_dataset_path, metric, "val")
            else:
                train_data = ContainerDataset(train_dataset_path, metric, )
                val_data = ContainerDataset(val_dataset_path, metric)

            if filter_model is not None or no_augmented:
                print("Filter Model")
                train_data.container.filter(
                    model=filter_model, no_augmented=no_augmented)
                val_data.container.filter(
                    model=filter_model, no_augmented=no_augmented)
    elif isinstance(train_dataset_path, ContainerDataset):
        train_data = train_dataset_path
        val_data = val_dataset_path
    elif isinstance(train_dataset_path, torch.utils.data.dataset.ConcatDataset):
        train_data = train_dataset_path
        val_data = val_dataset_path
    else:
        raise Exception(f"Unknown dataset type {type(train_dataset_path)}")

    return train_data, val_data


def train(
        train_dataset_path,
        val_dataset_path,
        metric: str,
        lr: float = 0.000005,
        epochs: int = 60,
        batch_size=16,
        weight_decay: float = 0.001,
        momentum: float = 0.25,
        store_path: str = None,
        silent=False,
        filter_model=None,
        no_augmented=False,
        full_train=False,
        budget=None,
        **args,
):
    if budget is not None:
        epochs = int(budget)
    print("Train Dataset Path: ", train_dataset_path)
    print("Val Dataset Path: ", val_dataset_path)
    print("Metric: ", metric)
    print("Learning Rate: ", lr)
    print("Epochs: ", epochs)
    print("Batch Size: ", batch_size)
    print("Weight Decay: ", weight_decay)
    print("Momentum: ", momentum)
    print("Store Path: ", store_path)
    print("Silent: ", silent)
    print("Filter Model: ", filter_model)
    print("No Augmented: ", no_augmented)
    print("Full Train: ", full_train)
    print("Budget: ", budget)
    print("Device: ", DEVICE)

    loss = nn.MSELoss()
    metric_fn = regression_metrics

    train_data, val_data = preload_data(
        train_dataset_path, val_dataset_path, metric, filter_model,
        no_augmented, full_train
    )

    print(f"Train Data: {len(train_data)}")
    print(f"Val Data: {len(val_data)}")

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=100
    )
    val_loader = DataLoader(
        val_data, batch_size=256, shuffle=False, num_workers=4, prefetch_factor=100
    )

    model = MLP(num_layers=3)
    model.to(DEVICE)
    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for e in range(1, epochs + 1):
        train_epoch(train_loader, model, loss, optimizer, e, silent=silent)
    targets, predictions, val_loss = validate(
        val_loader, model, e, loss, silent=silent)

    metrics = metric_fn(predictions, targets)
    metrics["loss"] = val_loss
    print("Validation Metrics:", metrics)

    if store_path is not None:
        makedirs(dirname(store_path), exist_ok=True)
        torch.save(model.state_dict(), store_path)

    return metrics


def smac_optimization(
        train_dataset_path: str,
        val_dataset_path: str,
        metric: str,
        store_path: str = None,
        lr=(0.0001, 0.1),
        epochs=(5, 80),
        batch_size=(16, 256),
        momentum=(0.1, 0.9),
        filter_model=None,
        no_augmented=False,
        n_trials: int = 100,
        silent=True,
        multi_fidelity=False,
):
    params = {
        "metric": [metric],
        "filter_model": filter_model,
        "no_augmented": no_augmented,
        "lr": lr,
        "batch_size": batch_size,
        "momentum": momentum,
    }
    if not multi_fidelity:
        params["epochs"] = epochs

    configspace = ConfigurationSpace(params)

    # Preload data
    train_data, val_data = preload_data(
        train_dataset_path, val_dataset_path, metric, filter_model, no_augmented
    )

    train_dataset_path = train_data
    val_dataset_path = val_data

    def smac_train(
            config: Configuration,
            seed: int = 0,
            store_path=None,
            full_train=False,
            budget=None
    ):
        print("========================================")
        print(config)

        metrics = train(
            train_dataset_path,
            val_dataset_path,
            silent=silent,
            store_path=store_path,
            full_train=full_train,
            **config,
            budget=budget,
        )
        print(metrics)
        return metrics["loss"]

    # Scenario object specifying the optimization environment
    if store_path is not None:
        output_directory = join(dirname(store_path), "smac3_output")
    else:
        store_path = "smac3_output"
    if not multi_fidelity:
        print("Using Single-Fidelity SMAC")
        scenario = Scenario(
            configspace, deterministic=True, n_trials=n_trials,
            output_directory=output_directory
        )
        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(
            scenario, smac_train,
        )
    else:
        print("Using Multi-Fidelity SMAC")
        scenario = Scenario(
            configspace, deterministic=True, n_trials=n_trials,
            output_directory=output_directory,
            min_budget=5,
            max_budget=35,
        )
        # We want to run five random configurations before starting the optimization.
        initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

        # Create our intensifier
        intensifier = Hyperband(scenario)

        # Create our SMAC object and pass the scenario and the train method
        smac = MFFacade(
            scenario,
            smac_train,
            initial_design=initial_design,
            intensifier=intensifier,
        )

    incumbent = smac.optimize()
    smac_train(incumbent, store_path=store_path, full_train=True)
    print("Final Settings:")
    print(incumbent)
    print("Stored at:", store_path)


