# Bridging the Gap: Unifying the Training and Evaluation of Neural Network Binary Classifiers

## To use a BtG loss in your research:

- Install the torch-btg package
```
pip install torch-btg
```

- Then import and use the desired loss in your Python code:
```
from torch_btg.loss import fb_loss
...
criterion = fb_loss(beta=1.0)
```

Project Webpage: [btg.yale.edu](https://btg.yale.edu)

[PDF Paper](https://btg.yale.edu/papers/Bridging_the_Gap_Unifying_the_Training_and_Evaluation_of_Neural_Network_Binary_Classifiers.pdf)

Citation:

```
@inproceedings{tsoi2022bridging,
  title         = {Bridging the Gap: Unifying the Training and Evaluation of Neural Network Binary Classifiers},
  author        = {Tsoi, Nathan and Candon, Kate and Li, Deyuan and Milkessa, Yofti and V{\'a}zquez, Marynel},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2022}
}
```

## Paper Results

The following instructions can be used to reproduce the results from the paper.

This project was tested using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on an Ubuntu 18.04 machine with an NVIDIA GPU.

## Setup

### Dependencies

- Clone this repository.

- Install docker with NVIDIA support on your host machine, if it is not already installed. You can do this manually or just run this command: 

  ```bash
  sudo apt install curl
  curl -L https://gist.githubusercontent.com/nathantsoi/e668e83f8cadfa0b87b67d18cc965bd3/raw/setup_docker.sh | sudo bash
  ```

### Build the container

```
./container build
```

### Enter the container

All of the following commands should be run inside the container. Enter the container before running commands, by calling `./container shell`. Note that multiple shells may be used at once by opening a new terminal and running `./container shell` again.

### Download Datasets

All datasets except the CocktailParty dataset can be downloaded via this script:

```
./scripts/download_data.sh
```

#### CocktailParty Dataset

We use a preprocessed version of the CocktailParty dataset that is suited for binary classification. To obtain this version of the dataset, run the `binary_cocktail_party.ipynb` notebook.

Run Jupyter Lab:

```
# enter the container
./container shell

# start the jupyter lab server
jupyter lab --ip 0.0.0.0 --no-browser --notebook-dir notebooks
```

Then, open your browser to [http://127.0.0.1:8888](http://127.0.0.1:8888) and follow the directions to run the `binary_cocktail_party.ipynb` notebook. Once the notebook completes, you will see a new file in the project directory, `data/cocktailparty/cp_binary.csv` and then you are ready to train the models.

## Train Models

Once you have cloned the repository, installed Docker with NVIDIA support, and acquired the datasets, train the models with:

```
./container shell
./scripts/train.sh
```

## View Results

The results can be viewed in the `results.ipynb` notebook.

Run Jupyter Lab:
```
# enter the container
./container shell

# start the jupyter lab server
jupyter lab --ip 0.0.0.0 --no-browser --notebook-dir notebooks
```
