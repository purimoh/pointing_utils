import scipy.signal as signal


standard_filter = {"filtername": "kaiser", "fc": 10, "rdb": 10, "width": 5}


def get_filter(filter_kwargs, Ts=0.01):
    if filter_kwargs["filtername"] == "kaiser":
        N, beta = signal.kaiserord(
            filter_kwargs["rdb"], filter_kwargs["width"] * 2 * Ts
        )
        taps = signal.firwin(N, filter_kwargs["fc"] * 2 * Ts, window=("kaiser", beta))
        b, a = taps, 1
    else:
        b, a = filter_kwargs["b"], filter_kwargs["a"]
    return b, a


def apply_filter(x_interpolated, b, a):
    return signal.filtfilt(b, a, x_interpolated)


def filter(x_interpolated, filter_kwargs, Ts=0.01):
    return apply_filter(x_interpolated, *get_filter(filter_kwargs, Ts=Ts))


def filter_standard(x_interpolated, Ts=0.01):
    return filter(x_interpolated, standard_filter, Ts=Ts)
