import argparse

def ratio_type(value):
    ratio = float(value)
    if ratio < 0 or ratio > 1:
        raise argparse.ArgumentTypeError(f"Ratio must be between 0 and 1, got {ratio}")
    return ratio
