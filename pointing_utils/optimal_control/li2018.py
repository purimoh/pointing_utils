from pointing_utils.optimal_control.phillis1985family import (
    Phillis1985Family,
    KLNotValidatedError,
    KLDidNotConvergeError,
)
from pointing_utils.optimal_control.lqg_ih import LQG_IH

import numpy
import matplotlib.pyplot as plt

from pointing_utils.optimal_control.SOFCstepper import UnstableClosedLoopSystemError


class Li2018(Phillis1985Family):
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
        seed=None,
        F=None,
        Y=None,
        G=None,
        Z=None,
        D=None,
        compute_K_L=False,
        verbose="no",
    ):
        self.F = F if F is not None else numpy.zeros(A.shape)
        self.Y = Y if Y is not None else numpy.zeros(B.shape)
        self.G = G if G is not None else numpy.zeros((A.shape[0], 1))
        self.Z = Z if G is not None else numpy.zeros(C.shape)
        self.D = D if D is not None else numpy.zeros((C.shape[0], 1))
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
            seed=seed,
            compute_K_L=compute_K_L,
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
            "converger_max_calls": 6,
        },
    ):
        # alias
        A = self.A
        B = self.B
        C = self.C
        Y = self.Y
        F = self.F
        G = self.G
        D = self.D
        Z = self.Z
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
            K, L, A, B, C, Y, F, G, D, Z, Q, R, U, N
        )

        if single:
            K, L, error = self.check_KL(
                K, L, Knorm, Lnorm, mode="scorer", verbose=verbose
            )
            return K, L, error
        try:
            K, L = self.check_KL(K, L, Knorm, Lnorm, mode="validator", verbose=verbose)
        except KLNotValidatedError:
            K, L = self.check_KL(
                K,
                L,
                Knorm,
                Lnorm,
                mode="converger",
                verbose=verbose,
                **converger_args,
            )
        return K, L

    def find_K_L_recursion(self, K_init, L_init, A, B, C, Y, F, G, D, Z, Q, R, U, N):
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

            Zbar = numpy.block(
                [
                    [
                        numpy.zeros((2 * n - K.shape[0], Z.shape[1])),
                        numpy.zeros((2 * n - K.shape[0], 2 * m - Z.shape[1])),
                    ],
                    [-K @ Z, numpy.zeros((K.shape[0], 2 * m - Z.shape[1]))],
                ]
            )

            V = numpy.block(
                [[Q + L.T @ R @ L, -L.T @ R @ L], [-L.T @ R @ L, L.T @ R @ L + U]]
            )

            P = self.LinRicatti(Abar, Gbar @ Gbar.T, Ybar, Fbar, Zbar)
            S = self.LinRicatti(Abar.T, V, Ybar.T, Zbar.T)

            Pnorm.append(numpy.linalg.norm(P))
            Snorm.append(numpy.linalg.norm(S))

            P11 = P[:n, :n]
            P22 = P[n:, n:]
            S11 = S[:n, :n]
            S22 = S[n:, n:]

            K = P22 @ C.T @ numpy.linalg.pinv(D @ D.T + Z @ P11 @ Z.T)
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
        Z = self.Z
        Q = self.Q
        R = self.R
        U = self.U
        L = self.L
        K = self.K
        dt = self.timestep

        u = -self.L @ xhat  # command

        if noise:
            dbeta = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(1, 1))
            dgamma = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(1, 1))
            domega = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(G.shape[1], 1))
            ddelta = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(1, 1))
            dxi = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(D.shape[1], 1))

            dx = (A @ x + B @ u) * dt + F @ x * dbeta + Y @ u * dgamma + G @ domega
            dy = C @ x * dt + Z @ x * ddelta + D @ dxi
            dxhat = (A @ xhat + B @ u) * dt + K @ (dy - C @ xhat * dt)

        else:
            dx = (A @ x + B @ u) * dt
            dy = C @ x * dt
            dxhat = (A @ xhat + B @ u) * dt + K @ (dy - C @ xhat * dt)

        cost = xhat.T @ U @ xhat + x.T @ Q @ x + u.T @ R @ u

        return {"dx": dx, "dxhat": dxhat, "dy": dy, "u": u, "cost": cost}


def main():
    # All the initializations of various parameters and matrices
    I = 0.25
    b = 0.2
    ta = 0.03
    te = 0.04

    a1 = b / (ta * te * I)
    a2 = 1 / (ta * te) + (1 / ta + 1 / te) * b / I
    a3 = b / I + 1 / ta + 1 / te
    bu = 1 / (ta * te * I)

    tau_one = 0.1
    tau_two = 0.05

    timestep = 0.01

    A = numpy.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, -a1, -a2, -a3]])

    B = numpy.array([[0, 0, 0, bu]]).reshape((-1, 1))
    n = A.shape[0]

    # A = numpy.eye(n) + timestep*A
    # B = timestep*B

    C_one = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    C_two = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    Z_one = 1 / 2.5 * numpy.array([[0.03, 0, 0, 0], [0, 0.06, 0, 0], [0, 0, 0.3, 0]])
    Z_two = 2.5 * Z_one
    D_one = 1 / 2.5 * numpy.array([[5e-4, 0, 0], [0, 5e-3, 0], [0, 0, 2.5e-2]])
    D_two = 2.5 * D_one

    Q = numpy.diag([1, 0.01, 0, 0])
    Q_star = numpy.kron(numpy.diag([1, 0, 0]), Q)
    R = numpy.array([[1e-4]])
    U = numpy.diag([1, 0.1, 0.01, 0])
    U_star = numpy.kron(numpy.diag([1, 0, 0]), U)

    F = numpy.zeros(A.shape)
    Y = 0.02 * B
    G = 0.03 * numpy.eye(n)

    A_star = numpy.block(
        [
            [A, numpy.zeros((n, n)), numpy.zeros((n, n))],
            [numpy.eye(n) / tau_one, -numpy.eye(n) / tau_one, numpy.zeros((n, n))],
            [numpy.eye(n) / tau_two, numpy.zeros((n, n)), -numpy.eye(n) / tau_two],
        ]
    )

    B_star = numpy.block([[B], [numpy.zeros((n, 1))], [numpy.zeros((n, 1))]])

    C_star = numpy.block(
        [
            [numpy.zeros(C_one.shape), C_one, numpy.zeros(C_one.shape)],
            [numpy.zeros(C_one.shape), numpy.zeros(C_one.shape), C_two],
        ]
    )

    F_star = numpy.block(
        [
            [F, numpy.zeros((n, n)), numpy.zeros((n, n))],
            [numpy.zeros((n, n)), numpy.zeros((n, n)), numpy.zeros((n, n))],
            [numpy.zeros((n, n)), numpy.zeros((n, n)), numpy.zeros((n, n))],
        ]
    )
    Y_star = numpy.block([[Y], [numpy.zeros((n, 1))], [numpy.zeros((n, 1))]])
    G_star = numpy.block([[G], [numpy.zeros(G.shape)], [numpy.zeros(G.shape)]])

    Z_star = numpy.block(
        [
            [numpy.zeros(Z_one.shape), Z_one, numpy.zeros(Z_one.shape)],
            [numpy.zeros(Z_one.shape), numpy.zeros(Z_one.shape), Z_two],
        ]
    )

    D_star = numpy.block(
        [
            [D_one, numpy.zeros((D_one.shape[0], D_two.shape[1]))],
            [numpy.zeros((D_two.shape[0], D_one.shape[1])), D_two],
        ]
    )

    # ============== using standard LQG infinite horizon solution
    model = LQG_IH(A_star, B_star, C_star, Q_star, R, None, None, G=G_star, D=D_star)
    K_init, L_init = model.K.T, model.L

    i = 0
    while True:
        try:
            model = Li2018(
                A_star,
                B_star,
                C_star,
                Q_star,
                R,
                U_star,
                K=K_init,
                L=L_init,
                Ac=None,
                Bc=None,
                Cc=None,
                seed=None,
                F=F_star,
                Y=Y_star,
                G=G_star,
                Z=Z_star,
                D=D_star,
                verbose="vv",
                compute_K_L=True,
            )
            break
        except (KLDidNotConvergeError, UnstableClosedLoopSystemError):
            i += 1
        if i == 4:
            raise KLDidNotConvergeError

    x_init = numpy.zeros((A_star.shape[0],))
    x_init[0] = -0.5

    model.reset(timestep, 3, n_trials=1, x_init=x_init)
    mov, u, cost = model.simulate(noise=True)


if __name__ == "__main__":
    main()
