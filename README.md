# Flower species classification
This repo contains the basic building blocks for creating an image classification model for recognizing flowers of various species. It exists as an extension of [this](https://www.tekna.no/kurs/maskinlaringsworkshop---python-36454/) machine learning workshop, and subsequent variants, hosted by [Tekna](https://www.tekna.no). The repo consists of three main components:

1. The slides used in the workshop
2. A fully working guide implemented as a Jupyter Notebook
3. This readme, which also substitutes as a guide for setting up the necessary environment

Additionally there is a version of the guide implemented as six stand-alone python scripts, one per step, for users not familiar with Jupyter Notebooks.

## Setup
The goal of this setup is creating an environment where we can run the code listed in the guide. To achieve this goal there are two necessary prerequisites:

- A working Python 3 environment with Tensorflow and Keras installed
- A folder containing the dataset structured [as we want it](#preparing-the-dataset)

Additionally there is a brief introduction to jupyter notebooks, for those wanting to run the guide.


### Configuring the environment
Setting up our environment means installing Python and all the packages we will be needing for this project. We are in this guide going to use [conda](https://conda.io) as an environment manager and [pip](https://pypi.org/project/pip/) as a package manager. There does however exist a wide variety of options out there, and as long as you are able to run the [sanity check](#environment-sanity-check) you should be good.

#### Installing conda
Conda is a package, dependency and environment manager for several languages, but in this project we will take advantage of the environment management capabilities. We will be using a version called Miniconda, which is installed by downloading and running a bash-script. Note that both the URL and the name of the script varies depending on your OS.

##### macOS and Linux
macOS: https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

Linux: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

```
$ wget <url> 
$ sh Miniconda3-latest-<OS>-x86_64.sh
```

Running the script will trigger a bunch of prompts, one of which is 

```
Do you wish the installer to prepend the Miniconda3 install location to PATH in your /home/esten/.bashrc ? [yes|no]
```

where we recommend you answer yes. Once the installer finishes Miniconda is installed, including both Python and pip ready for use.

##### Windows
The miniconda installer for windows can be downloaded from 
https://conda.io/en/latest/miniconda.html

Follow installation guide and when it is completed, open the "Anaconda Prompt" from your start menu. Where the linux guide uses a terminal to execute commands, the windows user will use this "Anaconda prompt"
. 

#### Creating the environment
We can create an environment with our newly installed conda installation using the command ```conda create```. We do, however, have to source the .bashrc (or .bash_profile for Mac users) file modified in the previous step. This is not nessecary for windows:

```
$ source .bashrc 
```
Next, we run the command that creates a new environment. Here we name it "ml" and give it the default python version 3.6.
```
$ conda create --name ml python=3.6
```
The new environment has to be activated.
###### Linux / macOS
```
$ source activate ml
```
###### Windows
```
$ conda activate ml
```
If everything went as intended the command line prompt should now be prefixed with the name of the environment.

```
(ml) $ .
```

#### Installing packages
The two main packages necessary for this project is Tensorflow and Keras, which can both be installed by pip. Additionally we will install matplotlib and PIL for interacting with, and specifically showing, images.

```
(ml) $ pip install tensorflow
(ml) $ pip install keras
(ml) $ pip install matplotlib
(ml) $ pip install pillow
```
<b>(Note: Users with a GPU could install tensorflow-gpu instead of tensorflow to greatly increase the training efficiency, but as it is not nessecary to complete this guide, we will stick with the default CPU version)</b>

We also recommend installing Jupyter to be able to run the guide as a notebook:

```
(ml) $ pip install jupyter
```

#### Environment sanity check
We can check that everything works as it should by importing the packages in Python:

```
(ml) $ python -c "import tensorflow" 
(ml) $ python -c "import keras"
```

If you are able to run these commands without anything failing horribly (warnings are OK!) you are all set up.

Note that whether or not this setup runs smoothly depends heavily on what already exists on your OS. Typical problems relate to image-specific libraries used by tensorflow. If you run into trouble you should get far by googling, or by sending me an email at esten@epimed.ai

### Preparing the dataset
The dataset we will be using consists of images of flowers, 17 species with 80 samples each, and was created by the Visual Geometry Group at the University of Oxford. For later convenience we want the dataset structured as follows:

- Two folders, called train and val, each containing
- 17 folders, one per species, each containing
- A set of images of of flowers of the given species

We will be using 65 images per species for training and 15 images per species for validation.

#### Downloading the dataset
The dataset can be downloaded and unzipped into this repo's root folder (remeber to ```git clone```it first) as follows:

```
(ml) $ wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
(ml) $ tar -xvzf 17flowers.tgz
```
or by downloading the zipped file from http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
and extracting the files into this repo's root folder

#### Restructuring
Originally, the dataset is structured by having the first 80 images belonging to the first species, the next 80 images to the second, and so on. The file ```restructure.py```, found in the repo, will build the structure we want if it is run in the same folder as ```jpg```. This step can also be done manually, see [Preparing the dataset](README.md#preparing-the-dataset) section for more info.

Make sure that the newly created folder is called ```flowers``` (this will happen automatically after running ```restructure.py```) and is placed in the same directory as this guide, i.e. root folder of this repo. When this is the case you are ready to go!

## Jupyter notebooks
[Jupyter notebook](http://www.jupyter.org) is a web application for running and sharing code and documentation in a user friendly and readable format. If you installed jupyter as defined [here](#installing-packages) you are all set up to start using notebooks. To start, run jupyter from the terminal (note: You should be in the folder where guide.ipynb is located, which should be the root folder of this repo):

```
(ml) $ jupyter notebook
```

If you open [http://localhost:8888](http://localhost:8888) you should see the file-structure of the folder where you run the command, and if any .ipynb-files (such as the guide) exists, simply click them to get started.