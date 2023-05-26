from pointing_utils.optimal_control.phillis1985family import (
    Phillis1985Family,
    KLDidNotConvergeError,
    KLNotValidatedError,
)
from pointing_utils.optimal_control.SOFCstepper import UnstableClosedLoopSystemError

import numpy
import matplotlib.pyplot as plt


class Qian2013(Phillis1985Family):
    def __init__(
        self,
        A,
        B,
        C,
        Q,
        R,
        U,
        K=None,
        L=None,
        Ac=None,
        Bc=None,
        Cc=None,
        F=None,
        Y=None,
        G=None,
        D=None,
        seed=None,
        verbose="vv",
        compute_K_L=False,
    ):
        self.F = F
        self.Y = Y
        self.G = G
        self.D = D

        super().__init__(
            A,
            B,
            C,
            Q,
            R,
            U,
            K,
            L,
            Ac=Ac,
            Bc=Bc,
            Hc=Cc,
            compute_K_L=compute_K_L,
            seed=seed,
            verbose=verbose,
        )

    @property  # in line with notation from Li 2018
    def C(self):
        return self.H

    def compute_kalman_matrices(
        self,
        N=20,
        K_init=None,
        L_init=None,
        single=False,
        verbose="vv",
        converger_args={
            "converger_N": 20,
            "converger_base": 1.5,
            "converger_max_calls": 2,
        },
    ):
        print("entering")
        # alias
        A = self.A
        B = self.B
        C = self.C
        Y = self.Y
        F = self.F
        G = self.G
        D = self.D
        Q = self.Q
        R = self.R
        U = self.U

        K = (
            K_init
            if K_init is not None
            else self.rng.normal(size=(A.shape[0], C.shape[0]))
        )
        L = (
            L_init
            if L_init is not None
            else self.rng.normal(size=(B.shape[1], A.shape[1]))
        )

        Knorm, Lnorm, K, L = self.find_K_L_recursion(
            K, L, A, B, C, Y, F, G, D, Q, R, U, N
        )

        # if not iterations:
        #     K, L = self.check_KL(K, L, Knorm, Lnorm, mode="scorer", verbose=verbose)

        if single:
            K, L, error = self.check_KL(
                K, L, Knorm, Lnorm, mode="scorer", verbose=verbose
            )
            return K, L, error

        try:
            K, L = self.check_KL(K, L, Knorm, Lnorm, mode="validator", verbose=verbose)
        except KLNotValidatedError:
            K, L = self.check_KL(
                K, L, Knorm, Lnorm, mode="converger", verbose=verbose, **converger_args
            )
        return K, L

    def find_K_L_recursion(self, K_init, L_init, A, B, C, Y, F, G, D, Q, R, U, N):
        Lnorm = []
        Knorm = []
        Pnorm = []
        Snorm = []

        L = L_init
        K = K_init

        for i in range(N):
            Lnorm.append(numpy.linalg.norm(L))
            Knorm.append(numpy.linalg.norm(K))

            n, m = A.shape
            Abar = numpy.block([[A - B @ L, B @ L], [numpy.zeros((n, m)), A - K @ C]])

            Ybar = numpy.block([[-Y @ L, Y @ L], [-Y @ L, Y @ L]])

            Fbar = numpy.block(
                [
                    [F, numpy.zeros((F.shape[0], 2 * m - F.shape[1]))],
                    [F, numpy.zeros((F.shape[0], 2 * m - F.shape[1]))],
                ]
            )

            Gbar = numpy.block(
                [[G, numpy.zeros((G.shape[0], D.shape[1]))], [G, -K @ D]]
            )

            V = numpy.block(
                [[Q + L.T @ R @ L, -L.T @ R @ L], [-L.T @ R @ L, L.T @ R @ L + U]]
            )

            P = self.LinRicatti(Abar, Gbar @ Gbar.T, Ybar, Fbar)
            S = self.LinRicatti(Abar.T, V, Ybar.T)

            Pnorm.append(numpy.linalg.norm(P))
            Snorm.append(numpy.linalg.norm(S))

            P11 = P[:n, :n]
            P22 = P[n:, n:]
            S11 = S[:n, :n]
            S22 = S[n:, n:]

            K = P22 @ C.T @ numpy.linalg.pinv(D @ D.T)
            L = numpy.linalg.pinv(R + Y.T @ (S11 + S22) @ Y) @ B.T @ S11

        return Knorm, Lnorm, K, L

    def step(self, x, xhat, noise=True):
        # Aliases
        A = self.A
        B = self.B
        C = self.C
        Y = self.Y
        F = self.F
        G = self.G
        D = self.D
        Q = self.Q
        R = self.R
        U = self.U
        L = self.L
        K = self.K
        dt = self.timestep

        u = -L @ xhat  # command

        if noise:
            dbeta = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(1, 1))
            dgamma = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(1, 1))
            domega = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(G.shape[1], 1))
            ddelta = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(1, 1))
            dxi = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(D.shape[1], 1))

            dx = (A @ x + B @ u) * dt + F @ x * dbeta + Y @ u * dgamma + G @ domega
            dy = C @ x * dt + D @ dxi
            dxhat = (A @ xhat + B @ u) * dt + K @ (dy - C @ xhat * dt)

        else:
            dx = (A @ x + B @ u) * dt
            dy = C @ x * dt
            dxhat = (A @ xhat + B @ u) * dt + K @ (dy - C @ xhat * dt)

        cost = xhat.T @ U @ xhat + x.T @ Q @ x + u.T @ R @ u

        return {"dx": dx, "dxhat": dxhat, "dy": dy, "u": u, "cost": cost}


class Qian2013_1D(Qian2013):
    pass


class Qian2013_3D(Qian2013):
    def plot_trajectories(self, ax=None):
        if ax is None:
            fig, axs = plt.subplots(nrows=1, ncols=3)

        for m in self.Mov[:, :, 0, 0].transpose(1, 0):
            axs[0].plot(self.time, m, "-")
        for m in self.Mov[:, :, 4, 0].transpose(1, 0):
            axs[1].plot(self.time, m, "-")
        for m in self.Mov[:, :, 8, 0].transpose(1, 0):
            axs[2].plot(self.time, m, "-")
        return axs


def main():
    I = 0.25
    b = 0.2
    ta = 0.03
    te = 0.04

    a1 = b / (ta * te * I)
    a2 = 1 / (ta * te) + (1 / ta + 1 / te) * b / I
    a3 = b / I + 1 / ta + 1 / te
    bu = 1 / (ta * te * I)

    timestep = 0.01

    A = numpy.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, -a1, -a2, -a3]])

    B = numpy.array([[0, 0, 0, bu]]).reshape((-1, 1))
    n = A.shape[0]

    C = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    D = numpy.array([[1e-2, 0, 0], [0, 1e-2, 0], [0, 0, 5e-2]])

    Q = numpy.diag([1, 0.01, 0, 0])
    R = numpy.array([[1e-4]])
    U = numpy.diag([1, 0.1, 0.01, 0])

    F = numpy.zeros(A.shape)

    Y = 0.08 * B
    G = 0.03 * numpy.diag([1, 0.1, 0.01, 0.001])

    D = D * 0.35
    G = G * 0.35

    i = 0
    while True:
        try:
            model = Qian2013_1D(
                A,
                B,
                C,
                Q,
                R,
                U,
                K=None,
                L=None,
                Ac=None,
                Bc=None,
                Cc=None,
                seed=None,
                F=F,
                Y=Y,
                G=G,
                D=D,
                verbose="vv",
            )
            break
        except (KLDidNotConvergeError, UnstableClosedLoopSystemError):
            i += 1
        if i == 4:
            raise KLDidNotConvergeError

    x_init = numpy.zeros((A.shape[0],))
    x_init[0] = -0.5
    model.reset(timestep, 3, n_trials=1, x_init=x_init)

    mov, u, cost = model.simulate(noise=True)


if __name__ == "__main__":
    main()
