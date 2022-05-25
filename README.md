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
- `STRATEGY` is one of:
  - `CAL` - Contrastive Active Learning [[2]](#2) 
  - `MAX_ENTROPY` [[3]](#3)
  - `RANDOM` - a baseline in which the next sample for labeling is chosen uniformly at random 
- `--al_stratified` results in the first sample for labeling being stratified 

### ***Data Augmentation***

#### ***Backtranslation [[4]](#4) ***

To augment the dataset using backtranslation, from the `data_augmentation` directory, run:
```bash
python backtranslation.py --run
```
To see the quality of augmented data such as redundant translations, NaN and Data Quality for Text Classification metric [[5]](#5) run:

```bash
python backtranslation.py --visualise
```

To convert augmented dataset into the training set, run:

```bash
python backtranslation.py --convert
```

#### ***Training***

To train with augmented data, run:

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

### ***Zero-Shot via Textual Entailment [[6]](#6) ***

```bash
python main.py --experiment zero-shot [--zs_var] [--visualise]
```

`--zs_var` results in the Zero-Shot variability experiment, to see how much the results differ for different but semantically similar class names 

## References

<a id="1">[1]</a> 
S. Ha, D. J. Marchetto, S. Dharur, and O. I. Asensio, “Topic classification of electric vehicle consumer experiences with transformer-based
deep learning,” en, Patterns, vol. 2, no. 2, p. 100 195, Feb. 2021, ISSN:
2666-3899. 

<a id="2">[2]</a> 
K. Margatina, G. Vernikos, L. Barrault, and N. Aletras, “Active
Learning by Acquiring Contrastive Examples,” arXiv:2109.03764 [cs],
Sep. 2021, arXiv: 2109.03764. [Online]. Available: http://arxiv.org/abs/2109.03764 (visited on 02/08/2022).

<a id="3">[3]</a> 
D. J. C. MacKay, “The Evidence Framework Applied to Classification
Networks,” en, Neural Computation, vol. 4, no. 5, pp. 720–736, Sep.
1992, ISSN: 0899-7667, 1530-888X. DOI: 10.1162/neco.1992.4
.5.720. [Online]. Available: https://direct.mit.edu/neco/article/4/5/720-736/5662 (visited on 03/10/2022).

<a id="4">[4]</a> 
R. Sennrich, B. Haddow, and A. Birch, “Improving Neural Machine
Translation Models with Monolingual Data,” in Proceedings of
the 54th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), Berlin, Germany: Association for
Computational Linguistics, Aug. 2016, pp. 86–96. DOI: 10.18653/v
1/P16-1009. [Online]. Available: https://aclanthology.org/P16-1009 (visited on 03/18/2022).

<a id="5">[5]</a> 
J. Li, “Automatic data quality evaluation for text classification,”
[Online]. Available: https://datacentricai.org/neurips21/papers/72_CameraReady_li_jiazheng_data_centric_ai_workshop.pdf (visited on 05/24/2022).

<a id="6">[6]</a> 
W. Yin, J. Hay, and D. Roth, “Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach,” arXiv:1909.00161
[cs], Aug. 2019, arXiv: 1909.00161. [Online]. Available: http://arxiv.org/abs/1909.00161 (visited on 03/16/2022).