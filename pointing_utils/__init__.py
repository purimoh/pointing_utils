__version__ = "0.0.2-dev0"


from pointing_utils.throughput import FittsModel, FittsModelNoW
from pointing_utils.segment import segment, package, Segmenter
from pointing_utils.interpolate import interpolate
from pointing_utils.filter import filter_standard, filter

from pointing_utils.optimal_control.SOFCstepper import (
    UnstableClosedLoopSystemError,
    SOFCStepper,
)
from pointing_utils.optimal_control.phillis1985family import (
    KLDidNotConvergeError,
    KLNotValidatedError,
    Phillis1985Family,
)
from pointing_utils.optimal_control.lqg_ih import LQG_IH
from pointing_utils.optimal_control.qian2013 import Qian2013
from pointing_utils.optimal_control.li2018 import Li2018
