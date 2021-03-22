The provided implementation of ADMM is tested on Python 3.6. There are two scripts, where the first one is «admm_toolbox.py» which implements the code needed to run a centralized or distributed ADMM based version of two classification problems (SVM and logistic regression), and two regression problems (LASSO and ridge). The second script, «admm.py», contains an example on how to run each problem. Also, note that we include two example datasets: the housing dataset(https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) dataset for regression purposes and the  banknote dataset (https://archive.ics.uci.edu/ml/datasets/banknote+authentication) for classification.

To run the problems, the following steps should be followed:

1. First of all, a Python 3.6+ environment is required. We strongly recommend using a virtualenv.

2. The environment requirements should be installed. To do so, PIP is a package management system used to install and manage software packages written in Python:

(a) pip install -r requirements.txt

3. Note that a required package is CVX optimization toolbox, that is only required to evaluate how good the solution given by ADMM is. The toolbox could be modified to not use this package, as it is only needed for visualization and debugging purposes.

4. Run the main script by using python admm.py. Note that this will solve all problems in a distributed and centralized way, and show the results through the screen. 

You can also run your own script for your concrete problem. In order to do so, prepare a python script that contains the following commands (for a centralized problem):

1. Load the ADMM solver using cent_solver = AdmmCentralized(data_in, data_out, problem), where data_in denotes the dataset features, data_out the regression target or labels, and problem is a string "ridge", "lasso", "svm" or "logistic" depending on the problem we want to solve. This function creates an instance of the AdmmCentralized class, which is used to optimize. Note that this class implements three functions x_update, y_update and z_update that correspond to ADMM equations.

2. We can train by using cent_solver.train(niter), where niter is the number of iterations of the solver. The train() method calls the different update methods implemented in the class.

3. The results can be seen by using cent_solver.plot()

For a distributed problem, the instructions are very similar:

1. Load the ADMM solver using dist_solver = AdmmDistributed(data_in, data_out, problem), where data_in denotes the dataset features, data_out the regression target or labels, and problem is a string "ridge", "lasso", "svm" or "logistic" depending on the problem we want to solve. Note that in this case, data_in and data_out must be lists of elements, where each element in the list contains the private dataset for each node; also note that the number of nodes is automatically set to the number of private datasets. This function creates an instance of the AdmmDistributed class, which is used to optimize. Note that this class implements three functions x_update, y_update and z_update that correspond to ADMM update equations.

2. We can train by using dist_solver.train(niter), where niter is the number of iterations of the solver. The train() method calls the different update methods implemented in the class.

3. The results can be seen by using dist_solver.plot()

Note that we provide four example problems for illustration purposes: the code is commented so that it facilitates incorporating other problems to the toolbox.

Please, if using this implementation, cite the original paper:

@book{boyd2011distributed,
  title={Distributed optimization and statistical learning via the alternating direction method of multipliers},
  author={Boyd, Stephen and Parikh, Neal and Chu, Eric},
  year={2011},
  publisher={Now Publishers Inc}
}
