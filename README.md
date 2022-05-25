# Methods for data and user efficient annotation for multi-label topic classification 

This repo contains code for my Master's Thesis which explored the  following methods:

- Active Learning
  - Contrastive Active Learning
  - Max Entropy

- Data Augmentation
  - Backtranslation 

- Zero-Shot Learning 
  - Textual Entailment Task

## Enviornment Setup

This project runs in a docker container. To start the containter
with a GPU access, run:
```bash
./scripts/start.sh --gpu [--detach]
```

If the container is built for the first time, you need to install pytorch from source:
 
```bash
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Project structure

The directory structure of the whole project is as follows:

```
.
├── data
├── notebooks
├── results
├── scripts
└── src
    ├── active_learning
    ├── config
    ├── data_augmentation
    ├── data_processing
    ├── few_shot
    ├── models
    └── visualisation
```
- `data` contains EV charging station reviews dataset with the original split to the train, test and validation. Data was collected in [[1]](#1) and made available [here](https://zenodo.org/record/4276350#.Yjh8mprMI-Q).

-  `notebooks` contains a jupyter notebook with data analysis 
- `results` contains pickle files and figures with experiments' results
- `scripts` contains code for setting up the enviornment
- `src` contains methods implementation. There is a separate directory for each method, additionally:
  - `config` contains parameter settigns for all experiments
  - `models` contains BART and BERT models used in different experiments
  - `visualisation` contains plotting utilities for all experiments

## Running Experiments

To run any experiment, go to `src` directory and use the following command:

```bash
 python main.py --experiment EXPERIMENT_NAME (EXPERIMENT_ARGS) [--visualise]
```

Where `EXPERIMENT_NAME` is one of: 
- `supervised` - train the model using the whole training set.
- `zero-shot` (Zero-Shot Learning) 
- `al` (Active Learning) 
- `augmentation` (Training with augmented data)
- `al_aug` (Using augmented data in an Active Learning loop) 

You can use --visualise to visualise the results of the experiment, those are saved to `results` directory and a corresponding subdirectory (e.g. results of zero-shot experiment will be saved in `results/zero-shot/figures`).

To see all available arguments, run:

```bash
python main.py --help
```

### ***Active Learning***
To run Active Learning experiment, use:

```bash
python main.py --experiment al --al_strategy STRATEGY [--al_stratified] [--visualise]
```

where:
- `STRATEGY` is one of: CAL, MAX_ENTROPY, RANDOM
- `--al_stratified` results in the first sample for labeling being stratified 

### ***Data Augmentation***

```bash
python main.py --experiment augmentation --aug_mode MODE [--visualise]
```
where `MODE` is:
- full - for the whole training set (+ corresponding augmentations) used for training
- small - for the limited training set. 
An additional argument can be provided which controls the number of samples used for training: `--data_size NUM_SAMPLES` 
This command will trigger two trainings: with and without augmented data.
- al_small - training with limited set which was selected during Active Learning iterations. 
This command will trigger Active Learning loop first (if there are no previous results) and
two trainings: with and without augmented data.

### ***Active Learning + Data Augmentation***

To use augmented data in Active Learning loop, run:

```bash
python main.py --experiment al_aug
```

Each AL selected sample is complemented with a corresponding augmented sample. 

### ***Zero-Shot via Textual Entailment***

```bash
python main.py --experiment zero-shot [--zs_var] [--visualise]
```

`--zs_var` results in the Zero-Shot variability experiment, to see how much the results differ for different but semantically similar class names 

## References

<a id="1">[1]</a> 
S. Ha, D. J. Marchetto, S. Dharur, and O. I. Asensio, “Topic classification of electric vehicle consumer experiences with transformer-based
deep learning,” en, Patterns, vol. 2, no. 2, p. 100 195, Feb. 2021, ISSN:
2666-3899. 


