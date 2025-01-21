import numpy as np

class EchoStateNetwork:
    def __init__(self, n_reservoir=500, spectral_radius=1.2, sparsity=0.1, noise=0.001):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise

    def fit(self, X, y):
        np.random.seed(42)
        self.W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        self.W *= self.sparsity
        rho = max(abs(np.linalg.eigvals(self.W)))
        self.W *= self.spectral_radius / rho
        self.W_in = np.random.rand(self.n_reservoir, X.shape[1]) - 0.5
        states = np.zeros((X.shape[0], self.n_reservoir))
        for t in range(1, X.shape[0]):
            states[t] = np.tanh(np.dot(self.W_in, X[t]) + np.dot(self.W, states[t-1])) + self.noise
        self.W_out = np.linalg.pinv(states).dot(y)
        return self

    def predict(self, X):
        states = np.zeros((X.shape[0], self.n_reservoir))
        for t in range(1, X.shape[0]):
            states[t] = np.tanh(np.dot(self.W_in, X[t]) + np.dot(self.W, states[t-1])) + self.noise
        return states.dot(self.W_out)
