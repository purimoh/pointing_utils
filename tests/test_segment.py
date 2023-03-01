from pointing_utils.segment import segment, package

import pathlib
import numpy


def test_exec():
    original_dataset_directory = pathlib.Path(__file__).resolve().parents[0]

    file = (original_dataset_directory / "traj_mueller.csv").resolve()
    with open(file, "r") as _file:
        for n, line in enumerate(_file):
            _words = line.split(",")

            if n == 0:
                print(
                    _words[24],
                    _words[29],
                    _words[30],
                    _words[31],
                    _words[32],
                    _words[33],
                )
                timestamps = []
                _width = []
                _dist = []
                _pos = []
                _spd = []
                _acc = []
            else:
                timestamps.append(float(_words[24]))
                _width.append(float(_words[29]))
                _dist.append(float(_words[30]))
                _pos.append(float(_words[27]))
                _spd.append(float(_words[32]))
                _acc.append(float(_words[33]))

        timestamps = numpy.array(timestamps)
        _pos = numpy.array(_pos)
        _spd = numpy.array(_spd)

        ## Cut first and last movement

        data_dict = {"x": _pos, "t": timestamps}
        movements, negmovs = segment(
            data_dict, reciprocal=True, resampling_period=0.01, trim=[1, -1]
        )

        fig, ax, handles = movements.plot_signals()

        (p2,) = ax.plot(
            movements.start_of_movements[:, 0],
            movements.start_of_movements[:, 1],
            "g*",
            label="start",
        )
        (p3,) = ax.plot(
            movements.final[:, 0], movements.final[:, 1], "*r", label="final"
        )

        # Negative

        (p5,) = ax.plot(
            negmovs.start_of_movements[:, 0],
            negmovs.start_of_movements[:, 1],
            "g*",
            label="start",
        )
        (p6,) = ax.plot(negmovs.final[:, 0], negmovs.final[:, 1], "*r", label="final")
        handles = handles + [p2, p3]

        ax.legend(handles=handles)
        # plt.show()

        container = package(movements, negmovs, reciprocal=True)


if __name__ == "__main__":
    test_exec()
