from os import makedirs
from os.path import join, exists, dirname
import argparse

from usam.training.train import smac_optimization, train

ALL_METRICS = [
    "iou_tiny",
    "iou_large",
    "iou_small",
    "iou_base_plus",
    "iou_dense_prompt",
    "unsupervised_iou",
    "model_gap",
    "task_gap",
    "prompt_gap",
]

TRAINING_PARAMETERS = {
    "COCO": {
        "default": {
            "batch_size": 189,
            "epochs": 36,
            "lr": 0.0087130695692,
            "momentum": 0.1639321491122
        },
        "iou_tiny": {
            "batch_size": 189,
            "epochs": 36,
            "lr": 0.0087130695692,
            "momentum": 0.1639321491122
        },
        "iou_large": {
            "batch_size": 189,
            "epochs": 35,
            "lr": 0.0087130695692,
            "momentum": 0.1639321491122
        }
    },
    "DAVIS": {
        "task_gap": {
            "batch_size": 158,
            "epochs": 55,
            "lr": 0.029406075389,
            "momentum": 0.7093735218048
        },
        "iou_base_plus": {
            "batch_size": 130,
            "epochs": 70,
            "lr": 0.0016774068549,
            "momentum": 0.8908015870842
        },
        "iou_small": {
            "batch_size": 193,
            "epochs": 70,
            "lr": 0.0024243301281,
            "momentum": 0.8148493215442
        },
        "iou_tiny": {
            "batch_size": 151,
            "epochs": 63,
            "lr": 0.0702736108729,
            "momentum": 0.3075366146863
        },
        "iou_large": {
            "batch_size": 232,
            "epochs": 42,
            "lr": 0.0164839636621,
            "momentum": 0.1234703782399
        },
        "model_gap": {
            "batch_size": 235,
            "epochs": 77,
            "lr": 0.0039678611351,
            "momentum": 0.1280011787627
        },
        "iou_dense_prompt": {
            "batch_size": 62,
            "epochs": 80,
            "lr": 0.0230261929437,
            "momentum": 0.2442788582406
        },
        "unsupervised_iou": {
            "batch_size": 118,
            "epochs": 57,
            "lr": 0.0343696361285,
            "momentum": 0.8852591752
        },
        "prompt_gap": {
            "batch_size": 154,
            "epochs": 40,
            "lr": 0.0008989134277,
            "momentum": 0.5382621589924
        },
    },
    "SAV": {
        "default": {
            "batch_size": 193,
            "epochs": 9,
            "lr": 0.0087130695692,
            "momentum": 0.1639321491122
        },
    },
    "ADE20k": {
        "default": {
            "batch_size": 106,
            "epochs": 79,
            "lr": 0.0013075788956,
            "momentum": 0.8303298246553
        },
        "task_gap": {
            "batch_size": 208,
            "epochs": 75,
            "lr": 0.0411114656149,
            "momentum": 0.2459095045924
        },
        "iou_tiny": {
            "batch_size": 106,
            "epochs": 79,
            "lr": 0.0013075788956,
            "momentum": 0.8303298246553
        },
        "prompt_gap": {
            "batch_size": 193,
            "epochs": 70,
            "lr": 0.0024243301281,
            "momentum": 0.8148493215442
        },
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train all predictors')
    parser.add_argument('--train-data', type=str, help='Train data')
    parser.add_argument('--val-data', type=str, help='Validation data')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--filter-model', type=str, default=None)
    parser.add_argument('--no-augmented', action='store_true',)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--metrics', nargs='+', default=ALL_METRICS)
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--skip-smac', action='store_true')
    parser.add_argument('--multi-fidelity', action='store_true')

    args = parser.parse_args()
    assert exists(args.train_data) or "sav_ALL_" in args.train_data, \
        f"Train data {args.train_data} does not exist"
    if args.val_data is None:
        args.val_data = args.train_data
    assert exists(args.val_data) or "sav_ALL_" in args.val_data, \
        f"Validation data {args.val_data} does not exist"
    if not exists(args.output_dir):
        makedirs(args.output_dir)

    return args


def main(
        train_data: str,
        val_data: str,
        output_dir: str,
        no_augmented: bool = False,
        n_trials: int = 100,
        metrics: list = None,
        filter_model: str = None,
        silent: bool = False,
        skip_smac: bool = False,
        multi_fidelity: bool = False
):
    for metric in metrics:
        output_file = join(output_dir, metric, f'model.pth')
        makedirs(dirname(output_file), exist_ok=True)
        print(output_file)
        if not exists(output_file):
            if not skip_smac:
                print(f"Optimizing {metric} predictor with HPO using SMAC")
                smac_optimization(
                    train_dataset_path=train_data,
                    val_dataset_path= val_data,
                    metric=metric,
                    store_path=output_file,
                    n_trials=n_trials,
                    no_augmented=no_augmented,
                    filter_model=filter_model,
                    silent=silent,
                    multi_fidelity=multi_fidelity
                )
            else:
                # Check if COCO, DAVIS, SAV, or ADE20k
                dataset = train_data
                if "coco" in train_data.lower():
                    dataset = "COCO"
                elif "davis" in train_data.lower():
                    dataset = "DAVIS"
                elif "sav" in train_data.lower():
                    dataset = "SAV"
                elif "ade20k" in train_data.lower():
                    dataset = "ADE20k"

                if dataset not in TRAINING_PARAMETERS:
                    raise Exception(f"Dataset {dataset} not supported! "
                                    f"Add parameters to TRAINING_PARAMETERS")

                if metric in TRAINING_PARAMETERS[dataset]:
                    param = TRAINING_PARAMETERS[dataset][metric]
                else:
                    param = TRAINING_PARAMETERS[dataset]["default"]

                print(f"Optimizing {metric} predictor with parameters", param)

                measures = train(
                        train_dataset_path=train_data,
                        val_dataset_path=val_data,
                        metric=metric,
                        lr=param["lr"],
                        epochs=param["epochs"],
                        batch_size=param["batch_size"],
                        momentum=param["momentum"],
                        store_path=output_file,
                        silent=False,
                        filter_model=filter_model,
                        no_augmented=no_augmented,
                        full_train=True,
                )
                print(f"Final measures for {metric}: {measures}")


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    print(args)
    # Run optimization
    main(**args.__dict__)
