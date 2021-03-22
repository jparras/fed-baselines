The main script to be used is named «main_federated_bayesian.py» and it contains the instructions to run VIRTUAL algorithm. We also provide a «requirements.txt» that specifies the different libraries needed for the environment on which it will be executed. To run the problem, the following steps should be followed:

1. First of all, a Python 3.6+ environment is required. We strongly recommend using a virtualenv.

2. The environment requirements should be installed. To do so, PIP is a package management system used to install and manage software packages written in Python:

(a) pip install -r requirements.txt

3. Open the script «main_federated_bayesian.py», and note that there is a dictionary of hyperparameters named «config». By changing the parameters, we can change our neural network architecture, the number of clients, and the rest of hyperparamenters that have an influence on the training procedure. We provide also an example on how to load a dataset by making use of the MNIST problem, please note that you need to invoque the prepare_dataset() method in order to obtain the data that is going to be send to each node.

4. Finally, the last call is to the run_experiment() method, which invoques all the functions needed for training. Note that this code makes an extensive use of Tensorflow Distributions, so all the computation and updates of distributions are done under the hood, which is a significant difference regarding the implementation provided for the centralized case. Also, note that the implementation heavily depends on the use of Deferred Tensors, which are a special class of Tensorflow Probability that updates the whole chain of tensors every time that one of the tensors of the chain changes. That is, if the parameters of a distribution changes, then all the distributions that depend on this one change as well.

5. In order to run the code with the default hyperparameters, execute python main_federated_bayesian.py

6. The final results will be stored in the following path: logs\

Please, if using this code, cite the original paper:
@article{corinzia2019variational,
  title={Variational federated multi-task learning},
  author={Corinzia, Luca and Beuret, Ami and Buhmann, Joachim M},
  journal={arXiv preprint arXiv:1906.06268},
  year={2019}
}
