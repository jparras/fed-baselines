import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

'''
In order to append / modify a problem, you should:
(a) Add a new target function and modify the target method
(b) Add its analytical solution and modify the solve_analytical method
(c) Add the new problem to AdmmCentralized class (if you want to use ADMM centralized), i.e., modify updates of x and z
(d) Add the new problem to AdmmDistributedAgent class (if you want to use ADMM distributed), i.e., modify updates of x and z
'''


class AdmmDistributed(object):  # Class that prepares data for distributed training
    '''
    Note that this class is a convenient way to test a distributed ADMM implementation. In real deployments, no agent has
    access to all data (as this class does) and hence, it is not possible to compute the global loss unless we split the
    regularizer term among all distributed agents. In a real deployment, also, the analytical solution is not available.
    Note that this function is provided just for illustration and testing purposes.
    '''
    def __init__(self, data_in, data_out, problem, lam=0.1, rho=10, grad_steps=10, grad_mu=0.1):
        if type(data_in) is not list or type(data_out) is not list:
            raise RuntimeError('Data must be provided as a list of numpy arrays per agent')
        if len(data_in) is not len(data_out):
            raise RuntimeError('Input and output data lists must have the same number of elements')
        self.na = len(data_in)
        self.problem = problem  # Problem to be solved
        # To store training values
        self.f = None  # To store function values (global)
        # ADMM parameters
        self.lam = lam  # Regularizer
        self.rho = rho  # Quadratic weight
        self.grad_steps = grad_steps  # Number of steps per iteration in gradient / subgradient method (if needed)
        self.grad_mu = grad_mu  # Step size in gradient / subgradient method (if needed)
        # Store global data
        self.global_data_in = np.vstack(data_in)
        self.global_data_out = np.hstack(data_out)
        self.global_data_out = self.global_data_out.reshape([self.global_data_out.size, 1])
        # Build ADMM agents
        self.agents = [AdmmDistributedAgent(data_in[i], data_out[i], self.global_data_in.shape[0], self.na,
                                            self.problem, lam=self.lam, rho=self.rho, grad_steps=self.grad_steps,
                                            grad_mu=self.grad_mu) for i in range(self.na)]
        # Analytical solution (for comparison purposes)
        self.fopt, self.xopt = self.solve_analytical()

    def function(self, x):  # Global target function (this function is only for illustration purposes)
        return target(self.global_data_in, self.global_data_out, self.lam, x, self.problem, z=None, na=1)

    def solve_analytical(self):
        return solve_analytical(self.global_data_in, self.global_data_out, self.lam, self.problem)

    def train(self, niter):  # Distributed ADMM training!
        for agent in self.agents:
            agent.initialize()  # Initialize local values
        self.f = []  # Initialize global f value
        for iter in range(niter):
            # Update x (locally)
            for agent in self.agents:
                agent.x_update(agent.x[-1], agent.y[-1], agent.z[-1])
            # Update z (globally!)
            sum_x = np.zeros_like(self.agents[0].x[-1])
            sum_y = np.zeros_like(self.agents[0].y[-1])
            for agent in self.agents:
                sum_x += agent.x[-1]
                sum_y += agent.y[-1]

            for agent in self.agents:
                agent.z_update(sum_x / self.na, sum_y / self.na, agent.z[-1])
            # Update y (locally)
            for agent in self.agents:
                agent.y_update(agent.x[-1], agent.y[-1], agent.z[-1])
            # Update global f: make use of z (global value, shared by all agents)
            self.f.append(self.function(self.agents[0].z[-1]))

    def plot(self):

        # Plot the losses using the global variable z and all the data
        plt.plot(10 * np.log10(np.square(np.array(self.f) - self.fopt) + np.finfo(float).eps), 'b', label='global')
        # Plot also the losses using the local terms, x and z (the actual values obtained: the gap is due to x != z)
        sum_f_local = np.sum(np.array([np.array(agent.f) for agent in self.agents]), axis=0)
        plt.plot(10 * np.log10(np.square(sum_f_local - self.fopt) + np.finfo(float).eps), 'r', label='local')
        plt.title('ADMM distributed global loss')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.legend(loc='best')
        plt.show()
        '''
        for i, agent in enumerate(self.agents):
            plt.plot(agent.f, label=str(i))
        plt.plot(self.f, label='Global value')
        plt.title('ADMM distributed function values: local and global')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.show()

        for i, agent in enumerate(self.agents):
            plt.plot(np.squeeze(np.array(agent.x)), label=str(i))
        plt.title('ADMM distributed x')
        plt.xlabel('Iteration')
        plt.ylabel('x value')
        plt.legend(loc='best')
        plt.show()

        for i, agent in enumerate(self.agents):
            plt.plot(np.squeeze(np.array(agent.y)), label=str(i))
        plt.title('ADMM distributed y')
        plt.xlabel('Iteration')
        plt.ylabel('y value')
        plt.legend(loc='best')
        plt.show()

        for i, agent in enumerate(self.agents):
            plt.plot(np.squeeze(np.array(agent.z)), label=str(i))
        plt.title('ADMM distributed z')
        plt.xlabel('Iteration')
        plt.ylabel('z value')
        plt.legend(loc='best')
        plt.show()

        for i, agent in enumerate(self.agents):
            plt.plot(10 * np.log10(
                np.sum(np.square(np.squeeze(np.array(agent.z)) - np.squeeze(np.array(agent.x))), axis=1) + np.finfo(
                    float).eps), label=str(i))
        plt.title('ADMM distributed x-z convergence')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.legend(loc='best')
        plt.show()
        '''

class AdmmDistributedAgent(object):
    def __init__(self, local_data_in, local_data_out, d_tot, na, problem, lam=0.1, rho=10, grad_steps=10, grad_mu=0.001):
        self.ndata = local_data_in.shape[0]  # Number of data points (local dataset)
        self.ndata_tot = d_tot  # Number of data points (global dataset)
        self.data_dim = local_data_in.shape[1]  # Number of features per data point
        self.data_in = local_data_in  # Feature matrix
        self.data_out = local_data_out.reshape([self.ndata, 1])  # Labels / regression targets
        self.na = na  # Number of agents cooperating
        self.problem = problem  # Problem to be solved
        # To store training values
        self.x = None  # To store x values (local)
        self.y = None  # To store y values (local)
        self.z = None  # To store z values (global)
        self.f = None  # To store function values (local)
        # ADMM parameters
        self.lam = lam  # Regularizer
        self.rho = rho  # Quadratic weight
        self.grad_steps = grad_steps  # Number of steps per iteration in gradient / subgradient method (if needed)
        self.grad_mu = grad_mu  # Step size in gradient / subgradient method (if needed)

    def function(self, x, z):  # Local target function
        return target(self.data_in, self.data_out, self.lam, x, self.problem, z=z, na=self.na, ntot=self.ndata_tot)

    def x_update(self, x, y, z):
        if self.problem is "lasso" or self.problem is "ridge":
            term1 = np.linalg.inv(2 / self.ndata_tot * self.data_in.T @ self.data_in + self.rho * np.eye(self.data_dim))
            term2 = 2 / self.ndata_tot * self.data_in.T @ self.data_out + self.rho * (z - y)
            xnew = term1 @ term2
        elif self.problem is "svm":  # In this case, we use a subgradient approach for the hinge function
            for it in range(self.grad_steps):
                d = np.diag(np.squeeze(self.data_out)) @ self.data_in
                term1 = -1 / self.ndata_tot * np.sum(d[np.squeeze(d @ x < 1), :], axis=0).reshape([self.data_dim, 1])
                x = x - self.grad_mu * (term1 + self.rho * (x - z + y))
            xnew = x
        elif self.problem is "logistic":  ## We use a gradient method for the logistic function
            for it in range(self.grad_steps):
                d = np.diag(np.squeeze(self.data_out)) @ self.data_in
                denominator = np.repeat(1 + np.exp(d @ x), self.data_dim, axis=1)
                term1 = -1 / self.ndata_tot * np.sum(d / denominator, axis=0).reshape([self.data_dim, 1])
                x = x - self.grad_mu * (term1 + self.rho * (x - z + y))
            xnew = x
        else:
            raise RuntimeError('Problem not recognized')
        self.x.append(xnew)

    def y_update(self, x, y, z):
        ynew = y + x - z
        self.y.append(ynew)
        # Update also the function value!
        self.f.append(self.function(x, z))

    def z_update(self, x, y, z):  # In this case, x and y are the average of local x and y values!!
        if self.problem is "lasso":
            q = x + y
            v = self.lam / (self.na * self.rho)
            znew = np.maximum(np.zeros_like(q), q - v) - np.maximum(np.zeros_like(q), - q - v)
        elif self.problem is "ridge" or self.problem is "svm" or self.problem is "logistic":
            znew = (x+y) * self.rho * self.na / (self.lam + self.rho * self.na)
        else:
            raise RuntimeError('Problem not recognized')
        self.z.append(znew)

    def initialize(self):
        # Initialize values
        self.x = []  # To store x values
        self.y = []  # To store y values
        self.z = []  # To store z values
        self.f = []  # To store target function values

        self.x.append(np.zeros((self.data_dim, 1)))
        self.y.append(np.zeros((self.data_dim, 1)))
        self.z.append(np.zeros((self.data_dim, 1)))
        self.f.append(self.function(self.x[-1], self.z[-1]))


class AdmmCentralized(object):
    def __init__(self, data_in, data_out, problem, lam=0.1, rho=10, grad_steps=10, grad_mu=0.001):
        self.ndata = data_in.shape[0]  # Number of data points
        self.data_dim = data_in.shape[1]  # Number of features per data point
        self.data_in = data_in  # Feature matrix
        self.data_out = data_out.reshape([self.ndata, 1])  # Labels / regression targets
        self.problem = problem  # Problem to be solved
        # To store training values
        self.x = None  # To store x values
        self.y = None  # To store y values
        self.z = None  # To store z values
        self.f = None  # To store function values
        # ADMM parameters
        self.lam = lam  # Regularizer
        self.rho = rho  # Quadratic weight
        self.grad_steps = grad_steps  # Number of steps per iteration in gradient / subgradient method (if needed)
        self.grad_mu = grad_mu  # Step size in gradient / subgradient method (if needed)
        # Analytical solution (for comparison purposes)
        self.fopt, self.xopt = self.solve_analytical()

    def function(self, x):  # Target function
        return target(self.data_in, self.data_out, self.lam, x, self.problem, z=None, na=1)

    def solve_analytical(self):
        return solve_analytical(self.data_in, self.data_out, self.lam, self.problem)

    def x_update(self, x, y, z):
        if self.problem is "lasso" or self.problem is "ridge":
            term1 = np.linalg.inv(2 / self.ndata * self.data_in.T @ self.data_in + self.rho * np.eye(self.data_dim))
            term2 = 2 / self.ndata * self.data_in.T @ self.data_out + self.rho*(z-y)
            xnew = term1 @ term2
        elif self.problem is "svm":  # In this case, we use a subgradient approach for the hinge function
            for it in range(self.grad_steps):
                d = np.diag(np.squeeze(self.data_out)) @ self.data_in
                term1 = -1 / self.ndata * np.sum(d[np.squeeze(d @ x < 1), :], axis=0).reshape([self.data_dim, 1])
                x = x - self.grad_mu * (term1 + self.rho * (x - z + y))
            xnew = x
        elif self.problem is "logistic":  ## We use a gradient method for the logistic function
            for it in range(self.grad_steps):
                d = np.diag(np.squeeze(self.data_out)) @ self.data_in
                denominator = np.repeat(1 + np.exp(d @ x), self.data_dim, axis=1)
                term1 = -1 / self.ndata * np.sum(d / denominator, axis=0).reshape([self.data_dim, 1])
                x = x - self.grad_mu * (term1 + self.rho * (x - z + y))
            xnew = x
        else:
            raise RuntimeError('Problem not recognized')
        self.x.append(xnew)

    def y_update(self, x, y, z):  # Always the same, we update the function value here!
        ynew = y + x - z
        self.y.append(ynew)
        # Update also the function value!
        self.f.append(self.function(x))

    def z_update(self, x, y, z):
        if self.problem is "lasso":
            q = x + y
            v = self.lam / self.rho
            znew = np.maximum(np.zeros_like(q), q - v) - np.maximum(np.zeros_like(q), - q - v)
        elif self.problem is "ridge" or self.problem is "svm" or self.problem is "logistic":
            znew = (x + y) * self.rho / (self.lam + self.rho)
        else:
            raise RuntimeError('Problem not recognized')
        self.z.append(znew)

    def initialize(self):
        self.x = []  # To store x values
        self.y = []  # To store y values
        self.z = []  # To store z values
        self.f = []  # To store target function values

        self.x.append(np.zeros((self.data_dim, 1)))
        self.y.append(np.zeros((self.data_dim, 1)))
        self.z.append(np.zeros((self.data_dim, 1)))
        self.f.append(self.function(self.x[-1]))

    def train(self, niter):  # Train centralized ADMM
        # Initialize values
        self.initialize()

        # Iterate ADMM
        for iter in range(niter):
            self.x_update(self.x[-1], self.y[-1], self.z[-1])  # Update x
            self.z_update(self.x[-1], self.y[-1], self.z[-1])  # Update z
            self.y_update(self.x[-1], self.y[-1], self.z[-1])  # Update y (and store the function value!)

    def plot(self):
        '''
        plt.plot(np.squeeze(np.array(self.x)), 'b', label='x')
        plt.plot(np.squeeze(np.array(self.y)), 'r', label='y')
        plt.plot(np.squeeze(np.array(self.z)), 'g', label='z')
        plt.title('ADMM centralized values')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.show()

        plt.plot(10 * np.log10(np.square(np.array(self.f))))
        plt.title('ADMM centralized function')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.show()
        '''

        plt.plot(10 * np.log10(np.square(np.array(self.f) - self.fopt) + np.finfo(float).eps))
        plt.title('ADMM centralized loss')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.show()


# Target functions (in distributed, x is local and z is global)
def target(data_in, data_out, lam, x, problem, z=None, na=1, ntot=None):
    if problem is "lasso":
        return target_lasso(data_in, data_out, lam, x, z, na=na, ntot=ntot)
    elif problem is "svm":
        return target_svm(data_in, data_out, lam, x, z, na=na, ntot=ntot)
    elif problem is "ridge":
        return target_ridge(data_in, data_out, lam, x, z, na=na, ntot=ntot)
    elif problem is "logistic":
        return target_logistic(data_in, data_out, lam, x, z, na=na, ntot=ntot)
    else:
        raise RuntimeError('Problem not recognized')


def target_lasso(data_in, data_out, lam, x, z, na=1, ntot=None):
    if ntot is None:
        ntot = data_in.shape[0]
    if z is None:
        return np.sum(np.square(data_in @ x - data_out)) / ntot + lam * np.sum(np.abs(x)) / na
    else:
        return np.sum(np.square(data_in @ x - data_out)) / ntot + lam * np.sum(np.abs(z)) / na


def target_ridge(data_in, data_out, lam, x, z, na=1, ntot=None):
    if ntot is None:
        ntot = data_in.shape[0]
    if z is None:
        return np.sum(np.square(data_in @ x - data_out)) / ntot + lam / 2 * np.sum(np.square(x)) / na
    else:
        return np.sum(np.square(data_in @ x - data_out)) / ntot + lam / 2 * np.sum(np.square(z)) / na


def target_svm(data_in, data_out, lam, x, z, na=1, ntot=None):
    if ntot is None:
        ntot = data_in.shape[0]
    cost = np.sum(np.maximum(np.zeros((data_in.shape[0], 1)),
                             np.ones((data_in.shape[0], 1)) - np.diag(np.squeeze(data_out)) @ (data_in @ x)))
    if z is None:
        return cost / ntot + lam / 2 * np.sum(np.square(x)) / na
    else:
        return cost / ntot + lam / 2 * np.sum(np.square(z)) / na


def target_logistic(data_in, data_out, lam, x, z, na=1, ntot=None):
    if ntot is None:
        ntot = data_in.shape[0]
    cost = np.sum(np.log(1 + np.exp(- np.diag(np.squeeze(data_out)) @ (data_in @ x))))
    if z is None:
        return cost / ntot + lam / 2 * np.sum(np.square(x)) / na
    else:
        return cost / ntot + lam / 2 * np.sum(np.square(z)) / na


# Analytical solutions to classical problems using CVX: this data can be later used for our problems as REFERENCE
def solve_analytical(data_in, data_out, lam, problem):
    if problem is "lasso":
        return solve_lasso_analytical(data_in, data_out, lam)
    elif problem is "svm":
        return solve_svm_analytical(data_in, data_out, lam)
    elif problem is "ridge":
        return solve_ridge_analytical(data_in, data_out, lam)
    elif problem is "logistic":
        return solve_logistic_analytical(data_in, data_out, lam)
    else:
        raise RuntimeError('Problem not recognized')


def solve_lasso_analytical(data_in, data_out, lam):
    x = cp.Variable((data_in.shape[1], 1))
    cost = cp.sum_squares(data_in @ x - data_out)
    prob = cp.Problem(cp.Minimize(cost/data_in.shape[0] + lam * cp.sum(cp.abs(x))))
    prob.solve()
    return prob.value, x.value


def solve_ridge_analytical(data_in, data_out, lam):
    x = cp.Variable((data_in.shape[1], 1))
    cost = cp.sum_squares(data_in @ x - data_out)
    prob = cp.Problem(cp.Minimize(cost/data_in.shape[0] + lam / 2 * cp.sum_squares(x)))
    prob.solve()
    return prob.value, x.value


def solve_svm_analytical(data_in, data_out, lam):
    x = cp.Variable((data_in.shape[1], 1))
    cost = cp.sum(cp.maximum(np.zeros((data_in.shape[0], 1)), np.ones((data_in.shape[0], 1)) - cp.diag(data_out) @ (data_in @ x)))
    prob = cp.Problem(cp.Minimize(cost/data_in.shape[0] + lam / 2 * cp.sum_squares(x)))
    prob.solve()
    return prob.value, x.value


def solve_logistic_analytical(data_in, data_out, lam):
    x = cp.Variable((data_in.shape[1], 1))
    cost = cp.sum(cp.logistic(-cp.diag(data_out) @ (data_in @ x)))
    prob = cp.Problem(cp.Minimize(cost / data_in.shape[0] + lam / 2 * cp.sum_squares(x)))
    prob.solve()
    return prob.value, x.value