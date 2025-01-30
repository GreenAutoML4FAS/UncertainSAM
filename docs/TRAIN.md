# Training

## Prepare additional Prerequisites

The following additional packages are required to run the training scripts.
Please install them using the following commands:

```shell
pip install pycocotools
pip install -U albumentations
pip install scikit-learn
```

If you want to use the hyperparameter optimization during training, you need to 
install SMAC using the following commands:

```shell
conda install gxx_linux-64 gcc_linux-64 swig
pip install smac
```

## Prepare Data

Prepare the data for training. The data should be stored in a directory structure
that is similar to the following example:

```
data
├── DAVIS
│   ├── Annotations
│   ├── ImageSets
│   ├── JPEGImages
│   └── SOURCES.md
├── SAV
│   ├── sav_test
│   │   ├── Annotations_6fps
│   │   ├── JPEGImages_24fps
│   │   └── sav_test.txt
│   ├── sav_train
│   │   ├── sav_000
│   │   ├── ...
│   │   └── sav_055
│   └── sav_val
│       ├── Annotations_6fps
│       ├── JPEGImages_24fps
│       └── sav_val.txt
├── ADE20k
│   └── ADEChallengeData2016
├── COCO
│   ├── val2017
│   ├── annotations
│   ├── images
│   ├── test2017
│   └── train2017
└── MOSE
    ├── meta_valid.json
    ├── meta_train.json
    ├── sample_submission_valid_all
    ├── train
    └── valid
```

**Note:**

- The DAVIS dataset can be downloaded from [here](https://davischallenge.org/davis2017/code.html#unsupervised). We used the full resolution dataset.
- If you have the download-link list for the SAV datasets, we provide the download script [download_sav.py](data/download_sav.py) that can be used to automatically download the datasets.



## Run scripts

Training is done in multiple steps.
First, we need to extract the latent tokens from the SAM model.
Then, we create a training container from the latent tokens.
Finally, we train the predictors.
The following commands show how to run the training scripts.

### 1. Extract latent Token from SAM

In the first step, we extract the latent tokens from the SAM model. During this 
step, also augmentations are applied to simulate aleatoric uncertainty estimation.
The augmentations are used in the paper to calculate the entropy measures, but
they are not necessary to train the USAM predictiors. Thus, to speed up the
token extraction, we can skip the augmentations by adding the `--skip-augs` flag.

```shell
python scripts/augment_dataset.py --dataset DAVIS --output-dir ${USAM_INFERENCE}/DAVIS --split val --model-root ${SAM_ROOT}/models/sam/checkpoints_2.0 --config-root ${SAM_ROOT}/models/sam/configs_2.0 --root ${SAM_ROOT}/data/DAVIS --skip-augs 
```

As result, in the `${USAM_INFERENCE}/DAVIS` directory, you will find a tree structure 
that is similar to the input data structure. The difference is that the images are
replaced by containers that contain the latent tokens and precomputed metrics.

### 2. Create Training Data Container

The next step is to create a training container from the extracted latent tokens.
This container is used to train the predictors. The training container is a
PyTorch dataset that can be used to train the predictors.
Run the following command to create the training container:

```shell
python scripts/create_training_container.py --root ${USAM_INFERENCE}/DAVIS --out-file ${USAM_INFERENCE}/DAVIS_training_container.pth 
```

If you have very large datasets and they should not be stored in memory during training,
you can load the main data to disk into a numpy file that is read from disk 
during training. To do so, you can add the `--offload` flag to the command.

### 3. Train Predictors

Finally, we can train the predictors. The following command trains all predictors.
The predictors are trained using the training container created in the previous step.
Per default, a hyperparameter optimization is performed using SMAC. If you want to
skip the hyperparameter optimization, you can add the `--skip-smac` flag.
Then, default parameters are used that were found to work well in the paper.

```shell
python scripts/train_all_predictors.py --train-data ${USAM_Inference}/DAVIS_training_container.pth --output-dir ${MODEL_OUTPUT_DIRECTORY}/output_models --skip-smac 
```


