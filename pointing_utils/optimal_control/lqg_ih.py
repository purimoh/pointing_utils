from pointing_utils.optimal_control.SOFCstepper import (
    SOFCStepper,
    UnstableClosedLoopSystemError,
)
import numpy
from control import care


class LQG_IH(SOFCStepper):
    def __init__(
        self,
        A,
        B,
        H,
        Q,
        R,
        K,
        L,
        G=None,
        D=None,
        Ac=None,
        Bc=None,
        Hc=None,
        seed=None,
        **kwargs
    ):
        self.G = G
        self.D = D
        super().__init__(A, B, H, Q, R, None, K, L, Ac, Bc, Hc, seed, **kwargs)
        self.K, self.L = self.compute_kalman_matrices()

    def compute_kalman_matrices(self):
        # alias
        A = self.Ac
        B = self.Bc
        H = self.Hc
        G = self.G
        D = self.D
        Q = self.Q
        R = self.R

        X, eigen, K = care(A.T, H.T, G @ G.T, D @ D.T)
        if (numpy.real(eigen) > 0).any():
            raise UnstableClosedLoopSystemError

        X, eigen, L = care(A, B, Q, R)
        if (numpy.real(eigen) > 0).any():
            raise UnstableClosedLoopSystemError

        return K, L

    def step(self, x, xhat, noise=True):
        # Aliases
        A = self.A
        B = self.B
        H = self.H
        G = self.G
        D = self.D
        Q = self.Q
        R = self.R
        L = self.L
        K = self.K
        dt = self.timestep

        u = -self.L @ xhat  # command

        if noise:
            domega = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(G.shape[1], 1))
            dxi = self.rng.normal(loc=0, scale=numpy.sqrt(dt), size=(D.shape[1], 1))

            dx = (A @ x + B @ u) * dt + G @ domega
            dy = C @ x * dt + D @ dxi
            dxhat = (A @ xhat + B @ u) * dt + K @ (dy - C @ xhat * dt)

        else:
            dx = (A @ x + B @ u) * dt
            dy = C @ x * dt
            dxhat = (A @ xhat + B @ u) * dt + K @ (dy - C @ xhat * dt)

        cost = x.T @ Q @ x + u.T @ R @ u

        return {"dx": dx, "dxhat": dxhat, "dy": dy, "u": u, "cost": cost}


def main():
    m = 100
    r = 300e3
    R = 6.37e6
    G = 6.67e-11
    M = 5.98e24
    k = G * M
    w = numpy.sqrt(k / (R + r) ** 3)
    v = w * (R + r)

    A = numpy.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [3 * w**2, 0, 0, 2 * (R + r) * w],
            [0, 0, -2 * w / (r + R), 0],
        ]
    )

    T = numpy.diag([1, r, 1, r])
    At = T @ A @ numpy.linalg.inv(T)

    B = numpy.array([[0, 0], [0, 0], [1 / m, 0], [0, 1 / (m * r)]])
    Bt = T @ B
    G = 0.1 * T @ B

    C = numpy.array([[0, 1, 0, 0]])
    Ct = C @ numpy.linalg.inv(T)

    D = numpy.array([[0.1]])
    Q = numpy.eye(4)
    R = numpy.eye(2)

    model = LQG_IH(At, Bt, Ct, Q, R, None, None, G=G, D=D)


if __name__ == "__main__":
    main()
