import matplotlib.pyplot as plt
import pandas
import numpy
import statsmodels.formula.api as smf
from emgregs import emg_reg_heterosked


class FittsModel:
    def __init__(self, dataframe, aggregate=["Participant", "A", "W"], throughputs=all):
        """
        Expects a dataframe with at least
        columns=["Participant", "X0", "Y0", "Xf", "Yf", "Xt", "Yt", "MT"]


        average = groupby argument for the aggregate metrics


        """

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

        self.aggregate_labels = aggregate
        # Add per-row effective distance, error, nominal ID IDn, error_date epsilon, standard deviation sigma, epsilon ID IDepsilon, effective ID IDe.
        self._augment()
        self._aggregate()

        if "meanofmeans" in throughputs or throughputs == "all":
            # compute mean of means values
            self._mean_of_means()
        if "slope" in throughputs or throughputs == "all":
            # Compute Linear regression
            self._slope_throughputs()
        if "emg" in throughputs or throughputs == "all":
            # Compute EMG regression
            self._emg_regression()

    def _aggregate(self):
        self.agg_df = (
            self.dataframe.drop(["X0", "Y0", "Xf", "Yf", "Xt", "Yt"], axis=1)
            .groupby(["Participant", "A", "W"])
            .mean()
        )

    def _emg_regression(self):

        a, b = emg_reg_heterosked(
            numpy.array(self.dataframe["IDn"]), numpy.array(self.dataframe["MT"])
        )["beta"]
        self.tp_data["EMG"] = 1 / b
        self._a_emg = a
        self._b_emg = b

        a, b = emg_reg_heterosked(
            numpy.array(self.agg_df["IDn"]), numpy.array(self.agg_df["MT"])
        )["beta"]
        self.tp_data["Agg-EMG"] = 1 / b
        self._a_emg_agg = a
        self._b_emg_agg = b

        a, b = emg_reg_heterosked(
            numpy.array(self.dataframe["IDe"]), numpy.array(self.dataframe["MT"])
        )["beta"]
        self.tp_data["EMG-effective"] = 1 / b
        self._a_emg_e = a
        self._b_emg_e = b

        a, b = emg_reg_heterosked(
            numpy.array(self.agg_df["IDe"]), numpy.array(self.agg_df["MT"])
        )["beta"]
        self.tp_data["Agg-EMG-effective"] = 1 / b
        self._a_emg_e_agg = a
        self._b_emg_e_agg = b

        a, b = emg_reg_heterosked(
            numpy.array(self.dataframe["IDepsilon"]), numpy.array(self.dataframe["MT"])
        )["beta"]
        self.tp_data["EMG-epsilon"] = 1 / b
        self._a_emg_eps = a
        self._b_emg_eps = b

        a, b = emg_reg_heterosked(
            numpy.array(self.agg_df["IDepsilon"]), numpy.array(self.agg_df["MT"])
        )["beta"]
        self.tp_data["Agg-EMG-epsilon"] = 1 / b
        self._a_emg_eps_agg = a
        self._b_emg_eps_agg = b

    def _slope_throughputs(self):
        a, b = FittsModel._compute_lr(self.dataframe["IDn"], self.dataframe["MT"])
        self._a_slope = a
        self._b_slope = b
        self.tp_data["Slope-Nominal"] = 1 / b

        a, b = FittsModel._compute_lr(self.agg_df["IDn"], self.agg_df["MT"])
        self._a_slope_agg = a
        self._b_slope_agg = b
        self.tp_data["Agg-Slope-Nominal"] = 1 / b

        a, b = FittsModel._compute_lr(self.dataframe["IDe"], self.dataframe["MT"])
        self.tp_data["Slope-Effective"] = 1 / b
        self._a_slope_e = a
        self._b_slope_e = b

        a, b = FittsModel._compute_lr(self.agg_df["IDe"], self.agg_df["MT"])
        self.tp_data["Agg-Slope-Effective"] = 1 / b
        self._a_slope_e_agg = a
        self._b_slope_e_agg = b

        a, b = FittsModel._compute_lr(self.dataframe["IDepsilon"], self.dataframe["MT"])
        self.tp_data["Slope-Epsilon"] = 1 / b
        self._a_slope_eps = a
        self._b_slope_eps = b

        a, b = FittsModel._compute_lr(self.agg_df["IDepsilon"], self.agg_df["MT"])
        self.tp_data["Agg-Slope-Epsilon"] = 1 / b
        self._a_slope_eps_agg = a
        self._b_slope_eps_agg = b

    def _mean_of_means(self):
        self.tp_data["Mean-of-Means"] = numpy.mean(
            self.agg_df["IDn"] / self.agg_df["MT"]
        )
        self.tp_data["ISO"] = numpy.mean(self.agg_df["IDe"] / self.agg_df["MT"])
        self.tp_data["Mean-of-Means-epsilon"] = numpy.mean(
            self.agg_df["IDepsilon"] / self.agg_df["MT"]
        )

    def _augment(self):
        self._add_line_per_line_to_df()
        self._add_line_per_line_agg_to_df()

    def _add_line_per_line_to_df(self):
        self.dataframe["effective_distance"] = FittsModel._effective_distance(
            self.dataframe
        )
        self.dataframe["error"] = FittsModel._detect_error(self.dataframe)
        self.dataframe["IDn"] = FittsModel._id(self.dataframe)

    def _add_line_per_line_agg_to_df(self):

        self.dataframe["epsilon"] = FittsModel._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, FittsModel._epsilon
        )
        self.dataframe["sigma"] = FittsModel._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, FittsModel._sigma
        )

        self.dataframe["IDepsilon"] = FittsModel._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, FittsModel._idepsilon
        )
        self.dataframe["IDe"] = FittsModel._broadcast_group_func_to_df(
            self.dataframe, self.aggregate_labels, FittsModel._ide
        )

    @staticmethod
    def _compute_lr(x, y):
        lm__out = smf.ols(formula="y~x", data=pandas.DataFrame({"x": x, "y": y})).fit()
        return lm__out.params

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
        return ((x["Xf"] - x["X0"]) ** 2 + ((x["Yf"] - x["Y0"]) ** 2)) ** (1 / 2)

    @staticmethod
    def _detect_error(x):
        return ((x["Xf"] - x["Xt"]) ** 2 + ((x["Yf"] - x["Yt"]) ** 2)) ** (1 / 2) > x[
            "W"
        ]

    @staticmethod
    def _sigma(x):
        # sigma is computed as the square root of the largest eigenvalue of the covariance matrix (spectral norm), see Gori, J. and Bellut, Q., Positional Variance Profiles (PVPs): A New Take on the Speed-Accuracy,CHI '23, April 23–28, 2023, Hamburg, Germany  for more information

        return (
            numpy.linalg.norm(
                numpy.cov(x["Xf"] - x["Xt"], x["Yf"] - x["Yt"], rowvar=False),
                ord=2,
            )
            ** (1 / 2)
        )

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
        FittsModel._regplot(
            ax,
            self.dataframe["IDn"],
            self._a_slope,
            self._b_slope,
            label="lr",
            color="r",
        )
        FittsModel._regplot(
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
        FittsModel._regplot(
            ax,
            self.agg_df["IDn"],
            self._a_slope_agg,
            self._b_slope_agg,
            label="lr",
            color="r",
        )
        FittsModel._regplot(
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
        FittsModel._regplot(
            ax,
            self.dataframe["IDe"],
            self._a_slope_e,
            self._b_slope_e,
            label="lr",
            color="r",
        )
        FittsModel._regplot(
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
        FittsModel._regplot(
            ax,
            self.agg_df["IDe"],
            self._a_slope_e_agg,
            self._b_slope_e_agg,
            label="lr",
            color="r",
        )
        FittsModel._regplot(
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
        FittsModel._regplot(
            ax,
            self.dataframe["IDepsilon"],
            self._a_slope_eps,
            self._b_slope_eps,
            label="lr",
            color="r",
        )
        FittsModel._regplot(
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
        FittsModel._regplot(
            ax,
            self.agg_df["IDepsilon"],
            self._a_slope_eps_agg,
            self._b_slope_eps_agg,
            label="lr",
            color="r",
        )
        FittsModel._regplot(
            ax,
            self.agg_df["IDepsilon"],
            self._a_emg_eps_agg,
            self._b_emg_eps_agg,
            label="emg",
            color="g",
        )
