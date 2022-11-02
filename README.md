# Bridging the Gap: Unifying the Training and Evaluation of Neural Network Binary Classifiers

Project Webpage: [btg.yale.edu](https://btg.yale.edu)

[PDF Paper](https://btg.yale.edu/papers/Bridging_the_Gap_Unifying_the_Training_and_Evaluation_of_Neural_Network_Binary_Classifiers.pdf)

Citation:

```
@article{tsoi2022btg,
  author        = {Tsoi, Nathan and Candon, Kate and Li, Deyuan and Milkessa, Yofti and V\'{a}zquez, Marynel},
  title         = {Bridging the Gap: Unifying the Training and Evaluation of Neural Network Binary Classifiers},
  journal       = {Advances in Neural Information Processing Systems},
  year          = {2022}
}
```

## Dependencies

This project was tested using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on an Ubuntu 18.04 machine with an NVIDIA  GPU.

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

Then, open your browser to [http://127.0.0.1:8888](http://127.0.0.1:8888) and run the `results.ipynb` notebook.

## Integrate with Your PyTorch Project

We welcome the use of our code under the included BSD 3-Clause "New" or "Revised" License.

To incorporate this code with your project:

- Copy the following two files into your project:

  - [confusion.py](src/confusion.py)

  - [btg.py](src/btg.py)

- Import the relevant helpers into your `main.py` or file with the training loop:

  ```
  from btg import mean_fbeta_approx_loss_on, mean_accuracy_approx_loss_on, mean_auroc_approx_loss_on
  ```

- Create an instance of your desired launch function. For example, for F1-Score: `criterion = mean_fbeta_approx_loss_on(device)` or for Accuracy: `criterion = mean_accuracy_approx_loss_on(device)`

- BtG losses expect network output in the [0,1] range. Typically, a Sigmoid activated final network layer should be used.

- Train as normal using criterion to compute your loss. Here is an abbreviated example from [the PyTorch docs](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network):
  ```
  for epoch in range(2):
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
  ```
