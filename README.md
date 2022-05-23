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
- `src` contains methods implementation. There is a separate folder for each method, additionally:
- `config` contains parameter settigns for all experiments
- `models` contains BART and BERT models used in different experiments
- `visualisation` contains plotting utilities for all experiments

## Running Experiments

To run any expoeriment, use the following command:

```bash
 python main.py --experiment EXPERIMENT_NAME (EXPERIMENT_ARGS)
```

Where `EXPERIMENT_NAME` is one of: 
- `zero-shot` (Zero-Shot Learning) 
- `al` (Active Learning) 
- `augmentation` (Training with augmented data)
- `al_aug` (Using augmented data in an Active Learning loop) 


## References

<a id="1">[1]</a> 
S. Ha, D. J. Marchetto, S. Dharur, and O. I. Asensio, “Topic classifi-
cation of electric vehicle consumer experiences with transformer-based
deep learning,” en, Patterns, vol. 2, no. 2, p. 100 195, Feb. 2021, ISSN:
2666-3899. 


