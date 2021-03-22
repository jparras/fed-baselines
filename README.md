# fed-baselines
Code prepared with several baselines to be used for federated learning problems. There are three different codes provided (see each folder for details):
 - admm folder contains an ADMM implementation, adequate for the case of convex problems, such as SVM, LASSO, logistic or ridge regressions.
 - BNNcent contains an implementation of Bayes By Backpropagation (BBB) algorithm, suitable to train a centralized Bayesian Neural Network efficiently.
 - BNNfed contains an implementation of VIRTUAL, an algorithm that builds upon BBB and Expectation-Propagation to train a distributed Bayesian Neural Network efficiently.
