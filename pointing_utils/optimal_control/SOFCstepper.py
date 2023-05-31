from abc import ABC, abstractmethod
import numpy

### This is Pouriya
class UnstableClosedLoopSystemError(Exception):
    """UnstableClosedLoopSystemError

    This error is raised if the computed K and L do not lead to a stable system (ie A-B@L or A - K@C have poles with positive real values).

    """

    pass


class SOFCStepper(ABC):
    def __init__(
        self, A, B, H, Q, R, U, K, L, Ac=None, Bc=None, Hc=None, seed=None, **kwargs
    ):
        """dx = (A @ x + B @ u)dt + Fx dnoise + Yu dnoise + G dnoise
        dy = H @ x*dt + D @ u*dt
        dxhat = (A_c @ xhat + B_c @ u)dt + K @ (dy - H_c @ xhat * dt)
        u = -L @ xhat

        costs: x.T @ Q @ x + u.T @ R @ u + (x-xhat).T @ U @ (x-xhat)

        """

        self.x = None

        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.U = U
        self.K = K
        self.L = L

        self.Ac = A if Ac is None else Ac
        self.Bc = B if Bc is None else Bc
        self.Hc = H if Hc is None else Hc

        self.rng = numpy.random.default_rng(seed=seed)

    def reset(self, timestep, simulation_time, x_init=None, n_trials=20):
        # initializes
        self.timestep = timestep
        self.simulation_time = simulation_time
        self.n_trials = n_trials
        TF = simulation_time  # alias

        time = [-timestep] + numpy.arange(0, TF, timestep).tolist()
        self.time = time

        # Mov.shape = (#timesteps, #trials, state, x or xhat)
        Mov = numpy.zeros((len(time), n_trials, self.A.shape[0], 2))
        if x_init is None:
            x_init = numpy.zeros((self.A.shape[0],))
            x_init[0] = self.rng.random()
        Mov[0, :, :, 0] = x_init  # initialize x
        Mov[0, :, :, 1] = x_init  # initialize xhat
        self.Mov = Mov
        self.u = numpy.zeros((len(time) - 1, n_trials, self.B.shape[1]))
        self.cost = numpy.zeros((n_trials,))

    @abstractmethod
    def step(self, x, xhat, noise=True):
        pass

    def simulate(self, noise=True):
        for nt in range(self.n_trials):
            for i, t in enumerate(self.time[1:]):
                x, xhat = self.Mov[i, nt, :, 0].reshape(-1, 1), self.Mov[
                    i, nt, :, 1
                ].reshape(-1, 1)
                step_result = self.step(x, xhat, noise=noise)
                dx, d_hat_x, u = (
                    step_result["dx"],
                    step_result["dxhat"],
                    step_result["u"],
                )
                self.Mov[i + 1, nt, :, 0] = (x + dx).reshape(1, -1)
                self.Mov[i + 1, nt, :, 1] = (xhat + d_hat_x).reshape(1, -1)
                self.u[i, nt, :] = u.squeeze()
                self.cost[nt] += step_result["cost"]

        return self.Mov, self.u, self.cost
