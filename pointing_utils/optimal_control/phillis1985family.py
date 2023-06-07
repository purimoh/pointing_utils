from pointing_utils.optimal_control.SOFCstepper import SOFCStepper
import numpy
import matplotlib.pyplot as plt


class KLDidNotConvergeError(Exception):
    """KLDidNotConvergeError

    This error is raised if the recursive computation of K and L does not converge.

    """

    pass


class KLNotValidatedError(Exception):
    """KLDidNotConvergeError

    This error is raised if the recursive computation of K and L does not converge.

    """

    pass


class Phillis1985Family(SOFCStepper):
    """Phillis1985Family

    class that builds on the Phillis 1985 paper formalism. Is used for the following papers:

        - Li, Zhe, et al. "A single, continuously applied control policy for modeling reaching movements with and without perturbation." Neural Computation 30.2 (2018): 397-427.
        - Qian, Ning, et al. "Movement duration, Fitts's law, and an infinite-horizon optimal feedback control model for biological motor systems." Neural computation 25.3 (2013): 697-724.
        - Gonzalez, Eric J., and Sean Follmer. "Sensorimotor Simulation of Redirected Reaching using Stochastic Optimal Feedback Control." Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems. 2023.


    :param OFCStepper: _description_
    :type OFCStepper: _type_
    :return: _description_
    :rtype: _type_
    """

    def __init__(
        self,
        A,
        B,
        H,
        Q,
        R,
        U,
        K,
        L,
        Ac=None,
        Bc=None,
        Hc=None,
        seed=None,
        compute_K_L=False,
        verbose="no",
    ):
        super().__init__(
            A,
            B,
            H,
            Q,
            R,
            U,
            K,
            L,
            Ac=Ac,
            Bc=Bc,
            Hc=Hc,
            seed=seed,
            compute_K_L=compute_K_L,
            verbose=verbose,
        )
        if L is not None:
            if L.shape != (B.shape[1], A.shape[0]):
                raise ValueError(
                    f"The L matrix does not have the right shape. Should be ({B.shape[1]}, {A.shape[0]}), but is {L.shape}"
                )
        if K is not None:
            if K.shape != (A.shape[0], H.shape[0]):
                raise ValueError(
                    f"The K matrix does not have the right shape. Should be {(A.shape[0],H.shape[0])}, but is {K.shape}"
                )
        if K is None or L is None:
            self.K, self.L = self.compute_kalman_matrices(verbose=verbose)

        if compute_K_L:
            self.K, self.L = self.compute_kalman_matrices(
                K_init=K, L_init=L, verbose=verbose
            )

    @staticmethod
    def LinRicatti(A, C, *args):
        # Compute the (L2) norm of the equation of the form AX + XA.T + (BXB.T for B in args) + C = 0
        n, m = A.shape
        nc, mc = C.shape
        if n != m:
            raise ValueError(
                "Matrix A has to be square, but is of shape {}x{}".format(*A.shape)
            )
        M = numpy.kron(numpy.identity(n), A) + numpy.kron(A, numpy.identity(n))
        for quad_matrix in args:
            M += numpy.kron(quad_matrix, quad_matrix)

        C = C.reshape(-1, 1)
        X = -numpy.linalg.solve(M, C)
        X = X.reshape(n, n)
        X = (X + X.T) / 2
        return X

    def counted_decorator(f):
        def wrapped(*args, **kwargs):
            if kwargs.get("mode") == "converger":
                wrapped.calls += 1
            return f(*args, **kwargs)

        wrapped.calls = 0
        return wrapped

    @counted_decorator
    def check_KL(
        self,
        K,
        L,
        Knorm,
        Lnorm,
        mode="validator",
        verbose="vv",
        converger_N=20,
        converger_base=1.5,
        converger_max_calls=6,
    ):
        average_delta = numpy.abs(
            numpy.convolve(
                numpy.diff(Lnorm) + numpy.diff(Knorm), numpy.ones(5) / 5, mode="full"
            )[-5]
        )
        print(f"mode = {mode}")
        print(f"check_kl_calls = {self.check_KL.calls}")
        if mode == "converger":
            if self.check_KL.calls == converger_max_calls:
                raise KLDidNotConvergeError()
            elif average_delta > 0.01:  # Arbitrary threshold
                if "v" in verbose:
                    print(
                        "--->>> the K and L matrices computations did not converge. Retrying with different starting point and a N={:d} search".format(
                            int(30 * converger_base**self.check_KL.calls)
                        )
                    )
                return self.compute_kalman_matrices(
                    N=int(converger_N * 1.5**self.check_KL.calls), verbose=verbose
                )
            else:
                return K, L
        elif mode == "validator":
            if "vv" in verbose:
                print("Knorm and Lnorm diffs")
                print(
                    numpy.abs(
                        numpy.convolve(
                            numpy.diff(Lnorm) + numpy.diff(Knorm),
                            numpy.ones(5) / 5,
                            mode="full",
                        )
                    )
                )
            if average_delta > 0.01:
                raise KLNotValidatedError()
            if not self.check_stability(K, L):
                raise SOFCStepper.UnstableClosedLoopSystemError()
            return K, L
        elif mode == "scorer":
            if not self.check_stability(K, L):
                return K, L, 1e5 + average_delta
            else:
                return K, L, average_delta

    def check_stability(self, K, L):
        feedback = self.A - self.B @ L
        feedback_real_part_poles = numpy.real(numpy.linalg.eig(feedback)[0])
        kalman_matrix = self.A - K @ self.C
        kalman_real_part_poles = numpy.real(numpy.linalg.eig(kalman_matrix)[0])

        return (feedback_real_part_poles < 0).all() and (
            kalman_real_part_poles < 0
        ).all()

    def plot_trajectories(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        for m in self.Mov[:, :, 0, 0].transpose(1, 0):
            ax.plot(self.time, m, "-")

        return ax
