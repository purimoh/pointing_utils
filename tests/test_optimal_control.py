from pointing_utils.optimal_control.lqg_ih import main as main_lqg
from pointing_utils.optimal_control.qian2013 import main as main_qian2013
from pointing_utils.optimal_control.li2018 import main as main_li2018


def test_lqg_ih():
    main_lqg()


def test_qian2013():
    main_qian2013()


def test_li2018():
    main_li2018()


if __name__ == "__main__":
    test_lqg_ih()
    test_qian2013()
    main_li2018()
