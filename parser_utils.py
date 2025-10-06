import argparse

def ratio_type(value):
    ratio = float(value)
    if ratio < 0 or ratio > 1:
        raise argparse.ArgumentTypeError(f"Ratio must be between 0 and 1, got {ratio}")
    return ratio


def positive_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"Random seed must be a positive integer, got {number}")
    return number