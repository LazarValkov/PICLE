# PICLE

This is a refactored version of the code used for "[A Probabilistic Framework for Modular Continual Learning](https://arxiv.org/abs/2306.06545)" [1].

#### Creating a conda environment: 
    conda create -n PICLE python=3.9
    conda activate PICLE
    pip install --upgrade pip
    pip install -r requirements.txt

#### Downloading the dataset

The underlying image datasets will be downloaded as needed when running the experiments.
The rest of the data is generated on the fly.

#### Running PICLE on a sequence from BELL
From the project's folder, running the following:

    python Experiments/BELL/Experiment.py -sequence S_out -cl_alg PICLE -device cpu -dbg
evaluates PICLE on the S<sup>out</sup> sequence of BELL on the cpu, for 1 training epoch per path, and 1 random seed.
To evaluate for all 3 random seeds and for more epochs, remove the -dbg argument.

This file can be used to run different baselines on different BELL sequences. To see all options, run:

    python Experiments/BELL/Experiment.py -h

#### Code readability note:
The code currently refers to paths as programs, as both specify how an input should be processed.

#### Rerefences
[1] Valkov, L., Srivastava, A., Chaudhuri, S. and Sutton, C., 2023. A Probabilistic Framework for Modular Continual Learning. arXiv preprint arXiv:2306.06545.