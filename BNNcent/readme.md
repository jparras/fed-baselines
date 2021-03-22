This code is based on https://github.com/saxena-mayur/Weight-Uncertainty-in-Neural-Networks and implements Bayes By Backpropagation algorithm for the MNIST classification problem.

There are two scripts implemented to carry out the classification problem, «bnn.py» and «bnn_toolbox.py», and «requirements.txt» that specifies the different libraries needed for the environment on which it will be executed. As a general overview, the code provides a simple interface designed to learn a probability distribution on the weights of a NN and allowing to modify the dataset used through a simple user implementation. The first script, «bnn.py», loads the MNIST dataset, preprocesses the data and initializes the model and its training; the second one, «bnn_toolbox.py», has the whole network implemented, with the steps to follow while training and evaluating the results.

To run the problem, the following steps should be followed:

1. First of all, a Python 3.6+ environment is required. We strongly recommend using a virtualenv.

2. The environment requirements should be installed. To do so, PIP is a package management system used to install and manage software packages written in Python:

(a) pip install -r requirements.txt

3. Talking about data, the main script, «bnn.py», downloads the MNIST data, which is already provided in the repository, and saves it both in raw and processed ways in the path data\MNIST\. There are several datasets provided in the «torchvision» library, which are subclasses of «Dataset» owning methods like __geitem__ and __len__. Hence, they can be passed to a «Dataloader», useful to load multiple samples in parallel and specifying the batch size or whether to shuffle the data or not, for example. It is easy enough, therefore, to change the dataset required by the user, since PyTorch facilities, such as the modules «Datasets» and «Dataloader», have been used. 

4. Finally, the main script should be run by setting the arguments desired.

(a) To check which arguments can be set: python bnn.py --help

(b) To run the code using a scale mixture of two Gaussian densities as the prior and 1200 hidden units (default), for example: python bnn.py

(c) To run the code using a Gaussian prior and 600 hidden units, for example: python bnn.py - -model gaussian_prior - -hidd_units 600

5. The final results (model, .csv file with metrics and figures) will be stored in the following path: results\

Please, if using this code, cite the original paper:

@inproceedings{blundell2015weight,
  title={Weight uncertainty in neural network},
  author={Blundell, Charles and Cornebise, Julien and Kavukcuoglu, Koray and Wierstra, Daan},
  booktitle={International Conference on Machine Learning},
  pages={1613--1622},
  year={2015},
  organization={PMLR}
}
