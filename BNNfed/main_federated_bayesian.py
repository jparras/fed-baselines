import os
from source.run_experiment import run_experiment
from source.data_utils import prepare_dataset
import numpy as np
import tensorflow as tf


if __name__=='__main__':

    # Generate the configuration parameters as a dictionary
    config = {"session":
                  {"num_gpus": 1,  # This is to run on GPU, note that if no GPU is detected, code runs on CPU!
                   "verbose": 1  # Controls the level of information returned
                   },
              "data_set_conf": {
                  "num_clients": 100  # Number of clients for the dataset
              },
              "training_conf": {
                  "method": "virtual",  # This uses VIRTUAL for training (other algos available are "fedprox")
                  "tot_epochs_per_client": 500,  # Optimization epochs per client
                  "fed_avg_init": False,
                  "tensorboard_updates": 1
              },
              "model_conf": {  # Layers of the NN model: note that the library supports conv and LSTM layers as well
                  "layers": [{
                      "input_shape": [784],
                      "name": "DenseReparametrizationNaturalShared",
                      "units": 100,
                      "activation": "relu",
                      "bias_posterior_fn": None
                  },
                      {
                          "name": "DenseReparametrizationNaturalShared",
                          "units": 100,
                          "activation": "relu",
                          "bias_posterior_fn": None
                      },
                      {
                          "name": "DenseReparametrizationNaturalShared",
                          "units": 10,
                          "activation": "softmax",
                          "bias_posterior_fn": None
                      }],
                  "hierarchical": False,
                  "prior_scale": 1.0
              },
              "hp": {  # Hyperparameters: note that by adding more elements to lists, it trains on a grid of hyperparameters!!
                  "epochs_per_round": [20],  # Epcohs per round
                  "natural_lr": [1e8],  # Used to modify the learning rate of the optimizer
                  "kl_weight": [1e-5],  # KL weight used (see paper)
                  "batch_size": [20],  # Batch size used for updates
                  "hierarchical": [False],  # Whether to use a hierarchical approach or not
                  "clients_per_round": [10],  # Number of clients updated per round
                  "learning_rate": [0.001],  # Learning rate
                  "optimizer": ["sgd"],  # optimizer chosen (sgd = stochastic gradient descent)
                  "scale_init": [[-4.85, 0.45]],  # Mean and variance of initial values scale (i.e., var or the params)
                  "loc_init": [[0, 0.5]],  # Mean and variance of initial values loc (i.e., mean or the params)
                  "server_learning_rate": [0.6]  # Learning rate of the server
              },
              "config_name": "mnist_virtual",
              "result_dir": os.getcwd()
              }

    # Load MNIST dataset (replace this part with your own dataset)

    num_clients = config["data_set_conf"]['num_clients']
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = np.split(x_train, 100)
    y_train = np.split(y_train, 100)
    x_test = np.split(x_test, 100)
    y_test = np.split(y_test, 100)

    federated_train_data, federated_test_data, train_size, test_size = \
        prepare_dataset(x_train, y_train, x_test, y_test, num_clients)

    num_clients = len(federated_train_data)
    config["model_conf"]['num_clients'] = num_clients

    # Run the code!! (pay attention to the input types to avoid errors in execution!)

    run_experiment(config, federated_train_data, federated_test_data, train_size, test_size)