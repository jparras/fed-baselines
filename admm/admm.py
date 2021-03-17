import numpy as np
from admm_toolbox import AdmmCentralized, AdmmDistributed


if __name__ == '__main__':

    np.random.seed(0)  # To make things repeatable
    problems = ["ridge", "lasso", "svm", "logistic"]
    niter = 1000

    for problem in problems:
        if problem is "ridge" or problem is "lasso":
            data = np.genfromtxt('housing.csv', delimiter=',')
            data = np.random.permutation(data)
            data_in = data[:, 0:-1]  # Data input
            data_out = data[:, -1]  # Regression target
            # Normalize data (improves convergence)
            ndata = data_in.shape[0]
            mean_in = np.vstack([np.mean(data_in, axis=0) for _ in range(ndata)])
            mean_out = np.mean(data_out)
            std_in = np.vstack([np.std(data_in, axis=0) for _ in range(ndata)])
            std_out = np.std(data_out)
            data_in = (data_in - mean_in) / std_in
            data_out = (data_out - mean_out) / std_out
        elif problem is "svm" or problem is "logistic":
            data = np.genfromtxt('banknote.csv', delimiter=',')
            data = np.random.permutation(data)
            data_in = data[:, 0:-1]  # Data input
            data_out = data[:, -1]  # Regression target
            data_out[data_out < 0.5] = -1  # Change from labels {0,1} to {-1, 1}
        else:
            raise RuntimeError('Problem not recognized')
        # Centralized ADMM
        cent_solver = AdmmCentralized(data_in, data_out, problem)
        cent_solver.train(niter)
        cent_solver.plot()

        # Distributed ADMM
        na = 2  # Number of ADMM agents
        dpa = int(data_in.shape[0] / na)  # Data points per agent
        d_in = []
        d_out = []
        for a in range(na):
            if a < na - 1:
                d_in.append(data_in[int(a*dpa):int((a+1)*dpa), :])
                d_out.append(data_out[int(a*dpa):int((a+1)*dpa)])
            else: # Last agent: take all the data available (in case data is not divided evenly)
                d_in.append(data_in[int(a * dpa):, :])
                d_out.append(data_out[int(a * dpa):])
        dist_solver = AdmmDistributed(d_in, d_out, problem)
        dist_solver.train(niter)
        dist_solver.plot()

    print('done')

