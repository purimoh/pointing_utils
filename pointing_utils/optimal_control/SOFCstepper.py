from abc import ABC, abstractmethod
from pointing_utils.optimal_control.phillis1985family import (
    Phillis1985Family,
    KLNotValidatedError,
    KLDidNotConvergeError,
)
import numpy
import matplotlib.pyplot as plt
import functools
from scipy.optimize import Bounds
import scipy.optimize as opti

class UnstableClosedLoopSystemError(Exception):
    """UnstableClosedLoopSystemError

    This error is raised if the computed K and L do not lead to a stable system (ie A-B@L or A - K@C have poles with positive real values).

    """

    pass

ntrials=20

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
####################### in the last line, I have added step redirection
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
                self.Mov[30:,:,0]+=2
        return self.Mov, self.u, self.cost

    def identify_A_B(
            reference_trajectory,
            timestep,
            TF,
            H,
            D,
            Ac,
            Bc,
            Hc,
            F,
            G,
            K,
            L,
            Y,
            noise="on",
            ntrials=ntrials,
            init_value=None,
    ):
        def my_distance(
                reference_trajectory,
                timestep,
                TF,
                H,
                D,
                Ac,
                Bc,
                Hc,
                F,
                G,
                K,
                L,
                Y,
                theta,
                noise=noise,
                ntrials=ntrials,
                init_value=None,
        ):
            a1, a2, a3, b1 = theta
            Anew, Bnew = build_matrices(a1, a2, a3, b1)
            Mov2, U2 = simulate_one_trajectory_with_built_matrices(
                timestep,
                TF,
                H,
                D,
                F,
                G,
                K,
                L,
                Y,
                Anew,
                Bnew,
                noise=noise,
                ntrials=ntrials,
                init_value=init_value,
            )
            mu = numpy.mean(reference_trajectory, axis=1)
            mu2 = numpy.mean(Mov2[:, :, 0, 0], axis=1)
            sig = numpy.power(numpy.std(reference_trajectory, axis=1), 2)
            sig2 = numpy.power(numpy.std(Mov2[:, :, 0, 0], axis=1), 2)
            err = sum((mu - mu2) ** 2 + 10 * ((sig - sig2) ** 2))
            return err

        opti_function = functools.partial(
            my_distance,
            reference_trajectory,
            timestep,
            TF,
            H,
            D,
            Ac,
            Bc,
            Hc,
            F,
            G,
            K,
            L,
            Y,
            noise=noise,
            ntrials=ntrials,
            init_value=init_value,
        )

        guess = [-10, -10, -5, 10]
        lb = [-numpy.inf, -numpy.inf, -numpy.inf, 0]
        ub = [0, 0, 0, numpy.inf]
        # ,method='powell', bounds=((None,0),(None,0),(None,0),(0,None))
        result_opt = opti.minimize(opti_function, guess, bounds=((None, 0), (None, 0), (None, 0), (0, None)))
        return result_opt


def build_matrices(a1, a2, a3, b1):
        Anew = numpy.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [0, a1, a2, a3]])

        Bnew = numpy.array([[0, 0, 0, b1]]).reshape((-1, 1))
        return Anew, Bnew
def simulate_one_trajectory_with_built_matrices(
    timestep,
    TF,
    H,
    D,
    F,
    G,
    K,
    L,
    Y,
    A,
    B,
    noise="on",
    ntrials=ntrials,
    init_value=None,
):
    return Phillis1985Family.plot_trajectories(
        timestep,
        TF,
        A,
        B,
        H,
        D,
        A,
        B,
        H,
        F,
        G,
        K,
        L,
        Y,
        noise=noise,
        ntrials=ntrials,
        init_value=init_value,
    )


