"""This file implements the TOSA elementwise binary operators."""


def add(input1, input2):
    """
    Elementwise add operator.

    :param input1: Input tensor
    :param input2: Input tensor with the same rank as input1
    :return: Output tensor with broadcast shape if necessary
    """

    if input1.ndim != input2.ndim:
        raise ValueError('input1.ndim != input2.ndim')

    return input1 + input2
