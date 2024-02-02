# Urban Sound Classification
This dataset contains 8732 sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.

# Getting started
In order to set up this project you will need the repository, and a virtual environment that contains all the required software dependencies.


## Installing GIT
Start by installing `GIT` on your system, which will allow us to clone the repository:
### Linux
Using apt (debian based): 
> sudo apt install git-all

Using dnf (RHEL based):

> sudo dnf install git-all

### MacOS
Use the homebrew package manager
> brew install git

### Windows
> Follow [this tutorial](https://git-scm.com/download/win) to set it up locally

Once git is installed, `cd` into the directory that you want this project to be located in and then clone this repository like so:

> git clone https://github.com/saptakdutta/audio_feature_extraction

You'll be prompted to enter in your gitlab username and password to clone the repo locally.
Now go ahead to the next part to set up the virtual environment

## Setting up the virtual environment
### Conda venv setup [`environment.yml` method] (wil add in a lockfile method later)
If you want to make changes to the repo itself and tinker around with the tool, using the environment file to create an up-to-date environment may be the better option.
Ensure that your conda install is upto date using:

> conda update conda

Use your python package manager (conda/miniconda/mamba) to cd into the root directory and run the following command:

> conda env create -f environment.yml

Now follow these steps to install pytorch and torchaudio in your virtual environment

First activate the vvirtual environment like so:

> conda activate sound_classification

Then run the following command:

> conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


This should create a virtual environment called `sound_classification` that will contain all the packages required to run this tool. I cannot guarantee that the environment will be resolved without conflicts (especially between the pip and conda packages). Some packages such as gensim and numba have been observed to create problems in the past. There may be a bit of tinkering with packages and versioning in the YML file that needs to be done to set the venv up correctly.

# Directory Structure
The local repo must have the following top level directrory layout: 

    .
    ├── /models
    ├── /data                    
    │   ├── /Test 
    │   ├── /Train
    │   ├── test.csv       
    │   └── train.csv
    ├── /predictions
    ├── main.py
    ├── Sound classification.ipynb
    ├── utils.py
    ├── environment.yml
    ├── todo.md
    ├── .gitignore        
    └── README.md         

Place the audio files to be analyzed into /audio_datasets/TargetEvent 