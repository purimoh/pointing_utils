import matplotlib.pyplot as plt
import numpy
import scipy.signal
import scipy.interpolate


class Segmenter:
    def __init__(
        self,
        data_dict,
        filter=None,
        resampling_period=0.01,
        compute_derivs=True,
        start_params={"thresh": 1e-2},
        stop_params={"thresh": 1e-2},
        trim=[0, 0],
    ):
        """__init__ _summary_

        filter is a dict with either {'taps': taps} or {'a':a, 'b':b}, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html


        :param data_dict: _description_
        :type data_dict: _type_json
        :param start_params: _description_, defaults to {"thresh": 1e-2}
        :type start_params: dict, optional
        :param trim: _description_, defaults to [0, 0]
        :type trim: list, optional
        """
        self.start_params = start_params
        self.trim = trim

        # interpolate, filter and potentially compute derivatives of the position signal
        if filter is None:
            # default kaiser window cutoff frequency at 10Hz
            # deviation ripple = 10dB
            # normalised width of transition region = 5
            N, beta = scipy.signal.kaiserord(10, 5 * 2 * resampling_period)
            taps = scipy.signal.firwin(
                N, 10 * 2 * resampling_period, window=("kaiser", beta)
            )
            filter = {"taps": taps}

        self.data_array = self._interp_filt(
            data_dict, resampling_period, filter, compute_derivs
        )

        # get rough midpoints for each movement
        self._midpoints_movements = Segmenter._find_movs(
            self.data_array[0, :], self.data_array[1, :], trim=trim
        )

        self.start_of_movements = Segmenter._find_start(
            self.data_array[0, :],
            self.data_array[1, :],
            self.data_array[2, :],
            self._midpoints_movements,
            **start_params
        )

        (
            self.first_zero,
            self.last_dwelling,
            self.score_based,
            self.final,
            self.mid,
        ) = Segmenter._find_stop(
            self.data_array[0, :],
            self.data_array[1, :],
            self.data_array[2, :],
            self._midpoints_movements,
            **stop_params
        )

    # https://matplotlib.org/stable/gallery/spines/multiple_yaxis_with_spines.html#sphx-glr-gallery-spines-multiple-yaxis-with-spines-py
    def _triplet_x(self, x, y1, y2, y3, fig=None, ax=None, labels=["y1", "y2", "y3"]):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        fig.subplots_adjust(right=0.75)

        twin1 = ax.twinx()
        twin2 = ax.twinx()
        # Offset the right spine of twin2.  The ticks and label have already been
        # placed on the right by twinx above.
        twin2.spines.right.set_position(("axes", 1.2))

        (p1,) = ax.plot(x, y1, "C0", label=labels[0])
        (p2,) = twin1.plot(x, y2, "C1", label=labels[1])
        (p3,) = twin2.plot(x, y3, "C2", label=labels[2])

        ax.set(xlabel="Time", ylabel="Position")
        twin1.set(ylabel="Speed")
        twin2.set(ylabel="Acceleration")

        ax.yaxis.label.set_color(p1.get_color())
        twin1.yaxis.label.set_color(p2.get_color())
        twin2.yaxis.label.set_color(p3.get_color())

        ax.tick_params(axis="y", colors=p1.get_color())
        twin1.tick_params(axis="y", colors=p2.get_color())
        twin2.tick_params(axis="y", colors=p3.get_color())

        handles = [p1, p2, p3]

        return fig, ax, twin1, twin2, handles

    def plot_signals(
        self, fig=None, ax=None, labels=["position", "speed", "acceleration"]
    ):
        fig, ax, _, _, handles = self._triplet_x(
            *self.data_array, labels=labels, fig=fig, ax=ax
        )
        return fig, ax, handles

    @staticmethod
    def _interp_filt(data_dict, resampling_period, filter, compute_derivs):
        x, y = data_dict["t"], data_dict["x"]
        x = numpy.array(x) - x[0]
        interp = scipy.interpolate.interp1d(x, y, fill_value="extrapolate")
        data_array = numpy.zeros((4, len(range(int(x[-1] / resampling_period + 1)))))
        data_array[0, :] = numpy.array(
            [resampling_period * i for i in range(int(x[-1] / resampling_period + 1))]
        )

        x_interp = interp(data_array[0, :])
        a = 1
        b = filter.get("taps", None)
        if b is None:
            a = filter.get("a")
            b = filter.get("b")

        data_array[1, :] = scipy.signal.filtfilt(b, a, x_interp)

        if compute_derivs:
            data_array[2, 1:] = numpy.divide(
                numpy.diff(data_array[1, :]), numpy.diff(data_array[0, :])
            )
            data_array[3, 1:] = numpy.divide(
                numpy.diff(data_array[2, :]), numpy.diff(data_array[0, :])
            )
        else:
            sig_interp = scipy.interpolate.interp1d(
                data_dict["v"], y, fill_value="extrapolate"
            )
            sig_interp = sig_interp(data_array[0, :])
            data_array[2, :] = scipy.signal.filtfilt(b, a, sig_interp)
            sig_interp = scipy.interpolate.interp1d(
                data_dict["a"], y, fill_value="extrapolate"
            )
            sig_interp = sig_interp(data_array[0, :])
            data_array[3, :] = scipy.signal.filtfilt(b, a, sig_interp)

        return data_array

    @staticmethod
    def _find_movs(_time, x, trim=[0, 0]):
        _movs = []
        _mean = 1 / 2 * (numpy.max(x) + numpy.min(x))
        _x, _y = [], []
        status = False
        for k, _test in enumerate(x > _mean):
            if _test == True and status == False:
                # Start of movement
                _x.append(_time[k])
                _y.append(x[k])

                status = True
            elif _test == True and status == True:
                # During Movement
                pass
            elif _test == False and status == False:
                # idle
                pass
            elif _test == False and status == True:
                # End of Movement
                status = False

        _x = _x[trim[0] :]
        _y = _y[trim[0] :]
        if trim[1] > 0:
            _x = _x[: -trim[1]]
            _y = _y[: -trim[1]]
        elif trim[1] < 0:
            _x = _x[: trim[1]]
            _y = _y[: trim[1]]

        return numpy.array([_x, _y])

    @staticmethod
    def _find_start_elem(_time, x, v, pt, previous_pt, thresh, mean_check=False):
        indx = numpy.where(_time == pt)
        indx = int(indx[0])
        mean = 1 / 2 * (numpy.max(x) + numpy.min(x))
        while abs(v[indx]) >= thresh:
            indx += -1

        if previous_pt > _time[indx]:
            return Segmenter._find_start_elem(
                _time, x, v, pt, previous_pt, thresh * 1.1
            )
        elif mean_check and x[indx] > mean:
            return Segmenter._find_start_elem(
                _time, x, v, pt, previous_pt, thresh * 1.1
            )
        else:
            return [_time[indx], x[indx], indx]

    @staticmethod
    def _find_start(_time, x, v, midpoints, thresh=1e-2):
        ### Start points
        startpt = []
        for k, pt in enumerate(midpoints[0]):
            if k == 0:
                pt_previous = 0
            else:
                pt_previous = midpoints[0][k - 1]
            try:
                startpt.append(
                    Segmenter._find_start_elem(_time, x, v, pt, pt_previous, thresh)
                )
            except RecursionError:
                startpt.append([None, None, None])

        return numpy.array(startpt)

    @staticmethod
    def _find_stop_elem(_time, x, v, pt, pt_after, thresh):
        # Score is based on the longest dwell period
        threshold = lambda signal: [x if abs(x) > thresh else 0 for x in signal]
        vthresh = threshold(v)
        mean = 1 / 2 * (numpy.max(x) + numpy.min(x))
        indx = numpy.where(_time == pt)
        indx = int(indx[0])
        # First zero crossing
        while vthresh[indx] != 0 and indx < (len(_time) - 1):
            indx += 1
        if indx == len(_time):
            return
        tmp_start = indx
        a = indx
        out = indx
        b = indx

        if _time[indx] > pt_after:
            return Segmenter._find_stop_elem(_time, x, v, pt, pt_after, thresh * 1.1)

        if x[indx] < mean:
            return Segmenter._find_stop_elem(_time, x, v, pt, pt_after, thresh * 1.1)

        first_zero = [_time[indx], x[indx], indx]
        if indx < (len(_time)):
            while x[indx] > mean and indx < (len(_time) - 1):
                indx += 1
        else:
            pass
        tmp_stop = indx
        plateau = vthresh[tmp_start:tmp_stop]
        Dwell = False
        plateaux = []
        for nu, u in enumerate(plateau[:-1]):
            if (
                u == 0
                and plateau[nu + 1] == 0
                and plateau[nu - 1] == 0
                and Dwell == False
            ):
                a = nu + tmp_start
                Dwell = True
                indx = a
            elif u != 0 and Dwell == True:
                b = nu + tmp_start
                Dwell = False
                plateaux.append([a, b])
            else:
                pass
        b = tmp_stop
        last_dwelling = [_time[a], x[a], indx]
        while abs(v[b]) > thresh and b > a + 1:
            b = b - 1
        final = [_time[b], x[b], b]
        _mid = int((b + a) / 2)
        mid = [_time[_mid], x[_mid], _mid]
        tmp = 0
        for a, b in plateaux:
            _dist = b - a
            if _dist > tmp:
                tmp = _dist
                out = a
        score_based = [_time[out], x[out], out]
        return first_zero, last_dwelling, final, mid, score_based

    @staticmethod
    def _find_stop(_time, x, v, midpoints, thresh=1e-2):
        ### Stop points

        first_zero = []
        last_dwelling = []
        score_based = []
        final = []
        mid = []  # # get rough midpoints for each movement

        for k, pt in enumerate(midpoints[0]):
            try:
                pt_after = midpoints[0][k + 1]
            except IndexError:
                pt_after = _time[-1]
            (
                _first_zero,
                _last_dwelling,
                _final,
                _mid,
                _score_based,
            ) = Segmenter._find_stop_elem(_time, x, v, pt, pt_after, thresh)

            first_zero.append(_first_zero)
            last_dwelling.append(_last_dwelling)
            mid.append(_mid)
            final.append(_final)
            score_based.append(_score_based)

        return (
            numpy.array(first_zero),
            numpy.array(last_dwelling),
            numpy.array(score_based),
            numpy.array(final),
            numpy.array(mid),
        )

    def check_segmentation(self):
        fig, ax, _ = self.plot_signals()
        ax.plot(*self._midpoints_movements, "b*")

        ax.plot(self.start_of_movements[:, 0], self.start_of_movements[:, 1], "g*")
        ax.plot(self.score_based[:, 0], self.score_based[:, 1], "r*")
        plt.show()

    def negative(self):
        self._midpoints_movements[:, 1] = -self._midpoints_movements[:, 1]
        self.start_of_movements[:, 1] = -self.start_of_movements[:, 1]
        self.final[:, 1] = -self.final[:, 1]
        self.first_zero[:, 1] = -self.first_zero[:, 1]
        self.last_dwelling[:, 1] = -self.last_dwelling[:, 1]
        self.score_based[:, 1] = -self.score_based[:, 1]
        self.mid[:, 1] = -self.mid[:, 1]
        return self


def update_thresholds(start_params, stop_params, reason):
    return {"thresh": start_params["thresh"] * reason}, {
        "thresh": stop_params["thresh"] * reason
    }


def segment(data_dict, segmenter=Segmenter, reciprocal=False, **kwargs):
    if not reciprocal:
        return segmenter(data_dict, **kwargs)
    else:
        positive = segmenter(data_dict, **kwargs)
        data_dict["x"] = -numpy.array(data_dict["x"])
        negative = segmenter(data_dict, **kwargs)
        negative = negative.negative()
        return positive, negative


def package(*args, reciprocal=False, json_serializable=False):
    if not reciprocal:
        segments = args[0]
        container = {}
        for n, (start, stop) in enumerate(
            zip(segments.start_of_movements[:, 2], segments.final[:, 2])
        ):
            if start is None or stop is None:
                continue
            start, stop = int(start), int(stop)
            if json_serializable:
                t = segments.data_array[0, start:stop].tolist()
                x = segments.data_array[1, start:stop].tolist()
            else:
                t = segments.data_array[0, start:stop]
                x = segments.data_array[1, start:stop]
            container["mov" + str(n)] = {
                "t": t,
                "x": x,
            }
    else:
        pos_segments, neg_segments = args
        container = {}
        for n, (start, stop) in enumerate(
            zip(pos_segments.start_of_movements[:, 2], pos_segments.final[:, 2])
        ):
            start, stop = int(start), int(stop)
            if json_serializable:
                t = pos_segments.data_array[0, start:stop].tolist()
                x = pos_segments.data_array[1, start:stop].tolist()
            else:
                t = pos_segments.data_array[0, start:stop]
                x = pos_segments.data_array[1, start:stop]

            container["mov" + str(n)] = {
                "t": t,
                "x": x,
            }
        k = n
        for n, (start, stop) in enumerate(
            zip(neg_segments.start_of_movements[:, 2], neg_segments.final[:, 2])
        ):
            start, stop = int(start), int(stop)
            if json_serializable:
                t = neg_segments.data_array[0, start:stop].tolist()
                x = neg_segments.data_array[1, start:stop].tolist()
            else:
                t = neg_segments.data_array[0, start:stop]
                x = neg_segments.data_array[1, start:stop]

            container["-mov" + str(n + k + 1)] = {
                "t": t,
                "x": x,
            }

    return container
