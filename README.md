# Temporal Audio VAE

This is the repo of the ATIAM ML project "Exploring the latent space of a temporal VAE for audio loops generation"

## Installation

```shell
pip install -r requirements.txt
```

## Usage

```shell
% cd src
% python -m temporal_audio_vae -h
usage: __main__.py [-h] --dbpath DBPATH [--checkpoint PATH] {train,gendataset,genrandom,genlatent} ...

positional arguments:
  {train,gendataset,genrandom,genlatent}

options:
  -h, --help            show this help message and exit
  --dbpath DBPATH
  --checkpoint PATH     continue from checkpoint
```

### Training

```shell
cd src
# with fixed beta:
python -m temporal_audio_vae --dbpath /path/to/data train --beta 0.2
# with beta warmup between epoch 10 and 20
python -m temporal_audio_vae --dbpath /path/to/data train --warmup 10 20 0 1
# continue from saved checkpoint:
python -m temporal_audio_vae --dbpath /path/to/data --checkpoint /path/to/checkpoint.pth train
```

Additional options are available:

```shell
% python -m temporal_audio_vae train -h                       
usage: __main__.py train [-h] [--beta BETA] [--warmup EPOCH_0 EPOCH_1 BETA_0 BETA_1] [--endepoch N] [--genepoch N] [--validepoch N] [--saveepoch N]

options:
  -h, --help            show this help message and exit
  --beta BETA           use fixed beta value
  --warmup EPOCH_0 EPOCH_1 BETA_0 BETA_1
                        use linear beta warmup from EPOCH_0 (beta=BETA_0) to EPOCH_1 (beta=BETA_1)
  --endepoch N          Stop at speicified epoch
  --genepoch N          generate every nth epoch
  --validepoch N        validate every nth epoch
  --saveepoch N         save every nth epoch
```

## Generation

```shell
# generate the 50th example from the training dataset:
python -m temporal_audio_vae --dbpath ../data/loops --checkpoint bla.pth gendataset 50

# generate from 5 random points in the latent space:
python -m temporal_audio_vae --dbpath ../data/loops --checkpoint bla.pth gendrandom 5

# explore the latent space and geneate 5 * n_latent sounds, by varying one dimension while keeping all the others fixed
python -m temporal_audio_vae --dbpath ../data/loops --checkpoint bla.pth gendrandom 5
```

## Repository structure

```bash
<repo-name>/
    biblio/ # bibliography and project subject
    config/ # directory to store the config files
    data/ # directory to store the data in local /!\ DO NOT COMMIT /!\
    docs/ # for github pages content
    models/ # directory to store checkpoints in local /!\ DO NOT COMMIT /!\
    notebooks/ # jupyter notebooks for data exploration and models analysis
    report/ # latex code of the report
    README.md # documentation
    requirements.txt # python project dependencies with versions
    .gitignore # indicate the files not to commit
    src/
        <package-name>/ # main package
            __init__.py
            datasets/ # data preprocessing and dataloader functions
            __init__.py
            helpers/ # global utility functions
            __init__.py
            models/ # models architecture defined as class objects
            __init__.py
    tests/ # tests package with unit tests
```
