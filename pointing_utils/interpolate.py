import numpy
import scipy.interpolate as interp


def interpolate(t, x, Ts=0.01):
    t = numpy.asarray(t)
    t = t - t[0]  # set null time target

    x = numpy.asarray(x)
    interpolator = interp.interp1d(t, x, fill_value="extrapolate")
    resampling_instants = numpy.linspace(
        0,
        t[-1] + (Ts - t[-1] % Ts),
        num=1 + int((t[-1] + (Ts - t[-1] % Ts)) / Ts),
    )
    return interpolator(resampling_instants)
