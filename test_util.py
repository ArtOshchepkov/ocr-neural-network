import numpy as np

from main import y_to_letter


def test_y_to_char():
    y = np.zeros(26)
    y[0] = 1.0

    char = y_to_letter(y)

    assert char == 'A'


def test_y_to_char_2():
    y = np.zeros(26)
    y[0] = 25.0
    y[1] = 1.5
    y[3] = 0.4

    char = y_to_letter(y)

    assert char == 'B'



def test_y_to_char_z():
    y = np.zeros(26)
    y[25] = 1.0

    char = y_to_letter(y)

    assert char == 'Z'
