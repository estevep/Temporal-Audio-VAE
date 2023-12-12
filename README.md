# Temporal Audio VAE

This is the repo of the ATIAM ML project "Exploring the latent space of a temporal VAE for audio loops generation"

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
