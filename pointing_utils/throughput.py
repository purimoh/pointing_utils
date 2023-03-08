import matplotlib.pyplot as plt
import pandas
import numpy
import statsmodels.api as sm
from emgregs import emg_reg_heterosked
from tabulate import tabulate
import scipy
from collections import namedtuple

FITTSMODELCOLUMNNAMES = ["Participant", "X0", "Y0", "Xf", "Yf", "Xt", "Yt", "MT"]


class FittsModel:
    def __init__(
        self,
        dataframe,
        aggregate=["Participant", "A", "W"],
        throughputs="all",
        CI=None,
        bootstrap_kwargs={"batch": None, "n_resamples": 9999},
    ):
        """
        Expects a dataframe with at least
        columns=["Participant", "X0", "Y0", "Xf", "Yf", "Xt", "Yt", "MT"]


        average = groupby argument for the aggregate metrics


        """

        self._ci = CI

        self.agg_df = None
        self.tp_data = {
            "Slope-Nominal": None,
            "Slope-Effective": None,
            "Slope-Epsilon": None,
            "Mean-of-Means": None,
            "ISO": None,
            "Mean-of-Means-epsilon": None,
            "EMG": None,
            "EMG-effective": None,
            "EMG-epsilon": None,
            "Agg-Slope-Nominal": None,
            "Agg-Slope-Effective": None,
            "Agg-Slope-Epsilon": None,
            "Agg-EMG": None,
            "Agg-EMG-effective": None,
            "Agg-EMG-epsilon": None,
        }

        # Load as dataframe
        if isinstance(dataframe, pandas.DataFrame):
            self.dataframe = dataframe
        else:
            self.dataframe = pandas.read_csv(dataframe)

        # self._check_dataframe_column_labels(self.dataframe)

        self.aggregate_labels = aggregate
        # Add per-row effective distance, error, nominal ID IDn, error_date epsilon, standard deviation sigma, epsilon ID IDepsilon, effective ID IDe.
        self._augment()
        self._aggregate()

        if "meanofmeans" in throughputs or throughputs == "all":
            # compute mean of means values
            self._mean_of_means(CI=CI, bootstrap_kwargs=bootstrap_kwargs)
        if "slope" in throughputs or throughputs == "all":
            # Compute Linear regression
            self._slope_throughputs(CI=CI, bootstrap_kwargs=bootstrap_kwargs)
        if "emg" in throughputs or throughputs == "all":
            # Compute EMG regression
            self._emg_regression(CI=CI)

    # def _check_dataframe_column_labels(self, dataframe):

    #     for key in dataframe.keys():
    #         if key not in self.__class__COLUMNNAMES:
    #             raise ValueError(
    #                 f'The provided dataframe should have columns named in ["Participant", "X0", "Y0", "Xf", "Yf", "Xt", "Yt", "MT"], but you provided {key}.'
    #             )

    def _aggregate(self):
        try:
            self.agg_df = (
                self.dataframe.drop(["X0", "Y0", "Xf", "Yf", "Xt", "Yt"], axis=1)
                .groupby(self.aggregate_labels)
                .mean()
            )
        except KeyError:
            self.agg_df = (
                self.dataframe.drop(["X0", "Xf", "Xt"], axis=1)
                .groupby(self.aggregate_labels)
                .mean()
            )

    def _emg_reg_nominal(self, CI=None):
        a, b = emg_reg_heterosked(
            numpy.array(self.dataframe["IDn"]), numpy.array(self.dataframe["MT"])
        )["beta"]
        self._a_emg = a
        self._b_emg = b
        if CI == "delta":
            self.tp_data["EMG"] = {"value": 1 / b, "CI": None}
        elif CI == "bootstrap":
            self.tp_data["EMG"] = {"value": 1 / b, "CI": None}
        else:
            self.tp_data["EMG"] = 1 / b

    def _emg_reg_nominal_agg(self, CI=False):
        a, b = emg_reg_heterosked(
            numpy.array(self.agg_df["IDn"]), numpy.array(self.agg_df["MT"])
        )["beta"]
        self._a_emg_agg = a
        self._b_emg_agg = b
        if CI == "delta":
            self.tp_data["Agg-EMG"] = {"value": 1 / b, "CI": None}
        elif CI == "bootstrap":
            self.tp_data["Agg-EMG"] = {"value": 1 / b, "CI": None}
        else:
            self.tp_data["Agg-EMG"] = 1 / b

    def _emg_reg_effective(self, CI=False):
        a, b = emg_reg_heterosked(
            numpy.array(self.dataframe["IDe"]), numpy.array(self.dataframe["MT"])
        )["beta"]
        self._a_emg_e = a
        self._b_emg_e = b
        if CI == "delta":
            self.tp_data["EMG-effective"] = {"value": 1 / b, "CI": None}
        elif CI == "bootstrap":
            self.tp_data["EMG-effective"] = {"value": 1 / b, "CI": None}
        else:
            self.tp_data["EMG-effective"] = 1 / b

    def _emg_reg_effective_agg(self, CI=False):
        a, b = emg_reg_heterosked(
            numpy.array(self.agg_df["IDe"]), numpy.array(self.agg_df["MT"])
        )["beta"]
        self._a_emg_e_agg = a
        self._b_emg_e_agg = b
        if CI == "delta":
            self.tp_data["Agg-EMG-effective"] = {"value": 1 / b, "CI": None}
        elif CI == "bootstrap":
            self.tp_data["EMG-effective"] = {"value": 1 / b, "CI": None}
        else:
            self.tp_data["Agg-EMG-effective"] = 1 / b

    def _emg_reg_epsilon(self, CI=False):
        a, b = emg_reg_heterosked(
            numpy.array(self.dataframe["IDepsilon"]), numpy.array(self.dataframe["MT"])
        )["beta"]
        self._a_emg_eps = a
        self._b_emg_eps = b
        if CI == "delta":
            self.tp_data["EMG-epsilon"] = {"value": 1 / b, "CI": None}
        elif CI == "bootstrap":
            self.tp_data["EMG-effective"] = {"value": 1 / b, "CI": None}
        else:
            self.tp_data["EMG-epsilon"] = 1 / b

    def _emg_reg_epsilon_agg(self, CI=False):
        a, b = emg_reg_heterosked(
            numpy.array(self.agg_df["IDepsilon"]), numpy.array(self.agg_df["MT"])
        )["beta"]
        self._a_emg_eps_agg = a
        self._b_emg_eps_agg = b
        if CI == "delta":
            self.tp_data["Agg-EMG-epsilon"] = {"value": 1 / b, "CI": None}
        elif CI == "bootstrap":
            self.tp_data["EMG-effective"] = {"value": 1 / b, "CI": None}
        else:
            self.tp_data["Agg-EMG-epsilon"] = 1 / b

    def _emg_regression(self, CI=False):

        self._emg_reg_nominal(CI=CI)
        self._emg_reg_nominal_agg(CI=CI)
        self._emg_reg_effective(CI=CI)
        self._emg_reg_effective_agg(CI=CI)
        self._emg_reg_epsilon(CI=CI)
        self._emg_reg_epsilon_agg(CI=CI)

    confint = namedtuple("ConfidenceInterval", ["low", "high", "standard_error"])

    @staticmethod
    def _get_ci_delta_method(b, standard_error):
        standard_error = 1 / b ** 2 * standard_error
        return FittsModel.confint(
            1 / b - 1.96 * standard_error, 1 / b + 1.96 * standard_error, standard_error
        )

    @staticmethod
    def _slope_tp(id, mt):
        covariates = numpy.ones((len(id), 2))
        covariates[:, 1] = id
        b = (
            numpy.linalg.inv((numpy.array(covariates).T @ numpy.array(covariates)))
            @ numpy.array(covariates).T
            @ mt
        )[1]

        return 1 / b

    def _bootstrap_utility(
        self, tpkey, dataframekey, idkey, value, tp_function, bootstrap_kwargs
    ):
        self.tp_data[tpkey] = {
            "value": value,
            "CI": scipy.stats.bootstrap(
                (getattr(self, dataframekey)[idkey], getattr(self, dataframekey)["MT"]),
                tp_function,
                paired=True,
                vectorized=False,
                **bootstrap_kwargs,
            ).confidence_interval,
        }

    def _delta_utility(self, tpkey, b, se):
        self.tp_data[tpkey] = {
            "value": 1 / b,
            "CI": FittsModel._get_ci_delta_method(b, se),
        }

    def _slope_tp_nominal(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        a, b, bse = self.__class__._compute_lr(
            self.dataframe["IDn"], self.dataframe["MT"]
        )
        self._a_slope = a
        self._b_slope = b
        if CI == "bootstrap":
            self._bootstrap_utility(
                "Slope-Nominal",
                "dataframe",
                "IDn",
                1 / b,
                FittsModel._slope_tp,
                bootstrap_kwargs,
            )
        elif CI == "delta":
            self._delta_utility("Slope-Nominal", b, bse[1])
        else:
            self.tp_data["Slope-Nominal"] = 1 / b

    def _slope_tp_nominal_agg(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        a, b, bse = self.__class__._compute_lr(self.agg_df["IDn"], self.agg_df["MT"])
        self._a_slope_agg = a
        self._b_slope_agg = b
        if CI == "bootstrap":
            self._bootstrap_utility(
                "Agg-Slope-Nominal",
                "agg_df",
                "IDn",
                1 / b,
                FittsModel._slope_tp,
                bootstrap_kwargs,
            )
        elif CI == "delta":
            self._delta_utility("Slope-Nominal", b, bse[1])
        else:
            self.tp_data["Agg-Slope-Nominal"] = 1 / b

    def _slope_tp_effective(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        a, b, bse = self.__class__._compute_lr(
            self.dataframe["IDe"], self.dataframe["MT"]
        )
        self._a_slope_e = a
        self._b_slope_e = b
        if CI == "bootstrap":
            self._bootstrap_utility(
                "Slope-Effective",
                "dataframe",
                "IDe",
                1 / b,
                FittsModel._slope_tp,
                bootstrap_kwargs,
            )
        elif CI == "delta":
            self._delta_utility("Slope-Effective", b, bse[1])
        else:
            self.tp_data["Slope-Effective"] = 1 / b

    def _slope_tp_effective_agg(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        a, b, bse = self.__class__._compute_lr(self.agg_df["IDe"], self.agg_df["MT"])
        self._a_slope_e_agg = a
        self._b_slope_e_agg = b
        if CI == "bootstrap":
            self._bootstrap_utility(
                "Agg-Slope-Effective",
                "agg_df",
                "IDe",
                1 / b,
                FittsModel._slope_tp,
                bootstrap_kwargs,
            )
        elif CI == "delta":
            self._delta_utility("Agg-Slope-Effective", b, bse[1])
        else:
            self.tp_data["Agg-Slope-Effective"] = 1 / b

    def _slope_tp_epsilon(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        a, b, bse = self.__class__._compute_lr(
            self.dataframe["IDepsilon"], self.dataframe["MT"]
        )
        self._a_slope_eps = a
        self._b_slope_eps = b
        if CI == "bootstrap":
            self._bootstrap_utility(
                "Slope-Epsilon",
                "dataframe",
                "IDepsilon",
                1 / b,
                FittsModel._slope_tp,
                bootstrap_kwargs,
            )
        elif CI == "delta":
            self._delta_utility("Slope-Epsilon", b, bse[1])
        else:
            self.tp_data["Slope-Epsilon"] = 1 / b

    def _slope_tp_epsilon_agg(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        a, b, bse = self.__class__._compute_lr(
            self.agg_df["IDepsilon"], self.agg_df["MT"]
        )
        self._a_slope_eps_agg = a
        self._b_slope_eps_agg = b
        if CI == "bootstrap":
            self._bootstrap_utility(
                "Agg-Slope-Epsilon",
                "agg_df",
                "IDepsilon",
                1 / b,
                FittsModel._slope_tp,
                bootstrap_kwargs,
            )
        elif CI == "delta":
            self._delta_utility("Agg-Slope-Epsilon", b, bse[1])
        else:
            self.tp_data["Agg-Slope-Epsilon"] = 1 / b

    def _slope_throughputs(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        # slope nominal
        self._slope_tp_nominal(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

        # Agg-Slope-Nominal
        self._slope_tp_nominal_agg(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

        # Slope effective
        self._slope_tp_effective(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

        # Agg-Slope-effective
        self._slope_tp_effective_agg(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

        # SLope-Epsilon
        self._slope_tp_epsilon(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

        # Agg-Slope-Epsilon
        self._slope_tp_epsilon_agg(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

    @staticmethod
    def _mm_tp(id, mt):
        return numpy.mean(id / mt)

    def _mm_nominal(self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}):
        if CI == "bootstrap":
            self._bootstrap_utility(
                "Mean-of-Means",
                "agg_df",
                "IDn",
                FittsModel._mm_tp(self.agg_df["IDn"], self.agg_df["MT"]),
                FittsModel._mm_tp,
                bootstrap_kwargs,
            )
        elif CI == "delta":
            self.tp_data["Mean-of-Means"] = {
                "value": FittsModel._mm_tp(self.agg_df["IDn"], self.agg_df["MT"]),
                "CI": None,
            }
        else:
            self.tp_data["Mean-of-Means"] = FittsModel._mm_tp(
                self.agg_df["IDn"], self.agg_df["MT"]
            )

    def _mm_effective(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        if CI == "bootstrap":
            self._bootstrap_utility(
                "ISO",
                "agg_df",
                "IDe",
                FittsModel._mm_tp(self.agg_df["IDe"], self.agg_df["MT"]),
                FittsModel._mm_tp,
                bootstrap_kwargs,
            )
        elif CI == "delta":
            self.tp_data["ISO"] = {
                "value": FittsModel._mm_tp(self.agg_df["IDe"], self.agg_df["MT"]),
                "CI": None,
            }
        else:
            self.tp_data["ISO"] = FittsModel._mm_tp(
                self.agg_df["IDe"], self.agg_df["MT"]
            )

    def _mm_epsilon(self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}):
        if CI == "bootstrap":

            self._bootstrap_utility(
                "Mean-of-Means-epsilon",
                "agg_df",
                "IDepsilon",
                FittsModel._mm_tp(self.agg_df["IDepsilon"], self.agg_df["MT"]),
                FittsModel._mm_tp,
                bootstrap_kwargs,
            )
        elif CI == "delta":
            self.tp_data["Mean-of-Means-epsilon"] = {
                "value": FittsModel._mm_tp(self.agg_df["IDepsilon"], self.agg_df["MT"]),
                "CI": None,
            }
        else:
            self.tp_data["Mean-of-Means-epsilon"] = FittsModel._mm_tp(
                self.agg_df["IDepsilon"], self.agg_df["MT"]
            )

    def _mean_of_means(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        self._mm_nominal(CI=CI, bootstrap_kwargs=bootstrap_kwargs)
        self._mm_effective(CI=CI, bootstrap_kwargs=bootstrap_kwargs)
        self._mm_epsilon(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

    def _augment(self):
        self._add_line_per_line_to_df()
        self._add_line_per_line_agg_to_df()

    def _add_line_per_line_to_df(self):
        self.dataframe["effective_distance"] = self.__class__._effective_distance(
            self.dataframe
        )
        self.dataframe["error"] = self.__class__._detect_error(self.dataframe)
        self.dataframe["IDn"] = self.__class__._id(self.dataframe)

    def _add_line_per_line_agg_to_df(self):

        self.dataframe["epsilon"] = self.__class__._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, self.__class__._epsilon
        )
        self.dataframe["sigma"] = self.__class__._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, self.__class__._sigma
        )

        self.dataframe["IDepsilon"] = self.__class__._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, self.__class__._idepsilon
        )
        self.dataframe["IDe"] = self.__class__._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, self.__class__._ide
        )

    @staticmethod
    def _compute_lr(x, y):
        x = sm.add_constant(x)
        lm__out = sm.OLS(y, x).fit()
        return *lm__out.params, lm__out.bse

    @staticmethod  # https://stackoverflow.com/questions/53747080/broadcast-groupby-result-as-new-column-in-original-dataframe
    def _broadcast_group_func_to_df(df, group_labels, func):
        return (
            df.groupby(group_labels, group_keys=False).apply(
                lambda x: pandas.Series(func(x), index=x.index).to_frame()
            )
        ).iloc[:, 0]

    @staticmethod
    def _epsilon(df):
        return df["error"].replace({True: 1, False: 0}).mean()

    @staticmethod
    def _ide(df):
        return numpy.log2(1 + df["effective_distance"].mean() / (4.133 * df["sigma"]))

    @staticmethod
    def _idepsilon(df):
        return (1 - df["epsilon"]) * numpy.log2(1 + df["A"] / df["W"])

    @staticmethod
    def _id(df):
        return numpy.log2(1 + df["A"] / df["W"])

    @staticmethod
    def _effective_distance(x):
        try:
            return ((x["Xf"] - x["X0"]) ** 2 + ((x["Yf"] - x["Y0"]) ** 2)) ** (1 / 2)
        except KeyError:
            return numpy.abs(x["Xf"] - x["X0"])

    @staticmethod
    def _detect_error(x):
        try:
            return ((x["Xf"] - x["Xt"]) ** 2 + ((x["Yf"] - x["Yt"]) ** 2)) ** (
                1 / 2
            ) > x["W"]
        except KeyError:
            return numpy.abs((x["Xf"] - x["Xt"])) > x["W"]

    @staticmethod
    def _sigma(x):
        # sigma is computed as the square root of the largest eigenvalue of the covariance matrix (spectral norm), see Gori, J. and Bellut, Q., Positional Variance Profiles (PVPs): A New Take on the Speed-Accuracy,CHI '23, April 23–28, 2023, Hamburg, Germany  for more information
        try:
            return (
                numpy.linalg.norm(
                    numpy.cov(x["Xf"], x["Yf"], rowvar=False),
                    ord=2,
                )
                ** (1 / 2)
            )
        except KeyError:
            return numpy.std(x["Xf"])

    @staticmethod
    def _regplot(ax, x, a, b, **regkwargs):
        xmin = min(x)
        xmax = max(x)
        ax.plot([xmin, xmax], [a + xmin * b, a + xmax * b], "-", **regkwargs)

    # ===== Plots ==== #
    def plot_fitts_ID_all(self, ax, reg=False, **kwargs):
        ax.plot(
            self.dataframe["IDn"],
            self.dataframe["MT"],
            "bo",
            **kwargs,
            label="MT vs ID",
        )
        if not reg:
            return
        self.__class__._regplot(
            ax,
            self.dataframe["IDn"],
            self._a_slope,
            self._b_slope,
            label="lr",
            color="r",
        )
        self.__class__._regplot(
            ax,
            self.dataframe["IDn"],
            self._a_emg,
            self._b_emg,
            label="emg",
            color="g",
        )

    def plot_fitts_ID_agg(self, ax, reg=False, **kwargs):
        ax.plot(
            self.agg_df["IDn"],
            self.agg_df["MT"],
            "bo",
            **kwargs,
            label="<MT> vs <ID>",
        )
        if not reg:
            return
        self.__class__._regplot(
            ax,
            self.agg_df["IDn"],
            self._a_slope_agg,
            self._b_slope_agg,
            label="lr",
            color="r",
        )
        self.__class__._regplot(
            ax,
            self.agg_df["IDn"],
            self._a_emg_agg,
            self._b_emg_agg,
            label="emg",
            color="g",
        )

    def plot_fitts_IDe_all(self, ax, reg=False, **kwargs):
        ax.plot(
            self.dataframe["IDe"],
            self.dataframe["MT"],
            "bo",
            **kwargs,
            label="MT vs IDe",
        )
        if not reg:
            return
        self.__class__._regplot(
            ax,
            self.dataframe["IDe"],
            self._a_slope_e,
            self._b_slope_e,
            label="lr",
            color="r",
        )
        self.__class__._regplot(
            ax,
            self.dataframe["IDe"],
            self._a_emg_e,
            self._b_emg_e,
            label="emg",
            color="g",
        )

    def plot_fitts_IDe_agg(self, ax, reg=False, **kwargs):
        ax.plot(
            self.agg_df["IDe"],
            self.agg_df["MT"],
            "bo",
            **kwargs,
            label="<MT> vs IDe",
        )
        if not reg:
            return
        self.__class__._regplot(
            ax,
            self.agg_df["IDe"],
            self._a_slope_e_agg,
            self._b_slope_e_agg,
            label="lr",
            color="r",
        )
        self.__class__._regplot(
            ax,
            self.agg_df["IDe"],
            self._a_emg_e_agg,
            self._b_emg_e_agg,
            label="emg",
            color="g",
        )

    def plot_fitts_IDepsilon_all(self, ax, reg=False, **kwargs):
        ax.plot(
            self.dataframe["IDepsilon"],
            self.dataframe["MT"],
            "bo",
            **kwargs,
            label=r"MT vs ID$(\varepsilon)$",
        )
        if not reg:
            return
        self.__class__._regplot(
            ax,
            self.dataframe["IDepsilon"],
            self._a_slope_eps,
            self._b_slope_eps,
            label="lr",
            color="r",
        )
        self.__class__._regplot(
            ax,
            self.dataframe["IDepsilon"],
            self._a_emg_eps,
            self._b_emg_eps,
            label="emg",
            color="g",
        )

    def plot_fitts_IDepsilon_agg(self, ax, reg=False, **kwargs):
        ax.plot(
            self.agg_df["IDepsilon"],
            self.agg_df["MT"],
            "bo",
            **kwargs,
            label=r"<MT> vs ID$(\varepsilon)$",
        )
        if not reg:
            return
        self.__class__._regplot(
            ax,
            self.agg_df["IDepsilon"],
            self._a_slope_eps_agg,
            self._b_slope_eps_agg,
            label="lr",
            color="r",
        )
        self.__class__._regplot(
            ax,
            self.agg_df["IDepsilon"],
            self._a_emg_eps_agg,
            self._b_emg_eps_agg,
            label="emg",
            color="g",
        )

    def summary(
        self,
        return_string=False,
        tabulate_kwargs={
            "tablefmt": "fancy_grid",
            "floatfmt": ".3f",
            "maxcolwidths": 10,
        },
    ):
        dic = {k: v for k, v in self.tp_data.items() if v is not None}
        if self._ci:
            tabu = tabulate(
                [
                    dic.keys(),
                    [f"{float(v['value']):.3f}" for v in dic.values()],
                    [
                        f"[{float(v['CI'].low):.3f}, {float(v['CI'].high):.3f}]"
                        if v["CI"] is not None
                        else ""
                        for v in dic.values()
                    ],
                ],
                headers="firstrow",
                **tabulate_kwargs,
            )
        else:
            tabu = tabulate(
                [dic.keys(), dic.values()], headers="firstrow", **tabulate_kwargs
            )
        if return_string:
            return tabu
        print(tabu)


class FittsModelNoW(FittsModel):
    def __init__(
        self,
        dataframe,
        aggregate=["Participant", "A"],
        throughputs="all",
        CI=False,
    ):
        super().__init__(dataframe, aggregate=aggregate, throughputs=throughputs, CI=CI)

    def _aggregate(self):
        try:
            self.agg_df = (
                self.dataframe.drop(["X0", "Y0", "Xf", "Yf"], axis=1)
                .groupby(self.aggregate_labels)
                .mean()
            )
        except KeyError:
            self.agg_df = (
                self.dataframe.drop(["X0", "Xf"], axis=1)
                .groupby(self.aggregate_labels)
                .mean()
            )

    def _emg_regression(self, CI=False):

        self._emg_reg_effective(CI=CI)
        self._emg_reg_effective_agg(CI=CI)

    def _slope_throughputs(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):
        # Slope effective
        self._slope_tp_effective(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

        # Agg-Slope-effective
        self._slope_tp_effective_agg(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

    def _mean_of_means(
        self, CI=False, bootstrap_kwargs={"batch": 4, "n_resamples": 9999}
    ):

        self._mm_effective(CI=CI, bootstrap_kwargs=bootstrap_kwargs)

    def _add_line_per_line_to_df(self):
        self.dataframe["effective_distance"] = self.__class__._effective_distance(
            self.dataframe
        )

    def _add_line_per_line_agg_to_df(self):

        self.dataframe["sigma"] = self.__class__._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, self.__class__._sigma
        )

        self.dataframe["IDe"] = self.__class__._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, self.__class__._ide
        )
