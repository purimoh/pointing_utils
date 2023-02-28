from pointing_utils.utils import FittsModel
import pathlib

original_dataset_directory = pathlib.Path(__file__).resolve().parents[0]



def test():
    print((original_dataset_directory / "fitts.csv").resolve())
    fm = FittsModel((original_dataset_directory / "fitts.csv").resolve())

if __name__ == '__main__':
    test()