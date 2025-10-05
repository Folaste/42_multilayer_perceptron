import argparse

from parser_utils import ratio_type
from split_data import split_data

if __name__ == "__main__":
    description = """
        Multilayer Perceptron is a program that can be used to train a neural network.
        You can use it to split the data, train the network, and predict the results.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-a", "--action",
                            type=str,
                            required=True,
                            help="Action to perform. Possible values: split, train, predict",
                            choices=["split", "train", "predict"]
                        )

    parser.add_argument("-d", "--dataset_path",
                            type=str,
                            required=False,
                            help="Path to the input dataset."
                        )

    parser.add_argument("-r", "--ratio",
                            type=ratio_type,
                            required=False,
                            help="Ratio of the validation set to the total dataset (between 0 and 1, default = 0.2).",
                            default=0.2
                        )

    parser.add_argument("-s", "--random_seed",
                            type=int,
                            required=False,
                            default=None,
                            help="Random seed to use for shuffling data and to initialize weights and biases in training program (default = None)."
                        )

    parser.add_argument("-n", "--normalisation_method",
                            type=str,
                            required=False,
                            help="Normalisation method to use for training dataset (default = z-score).",
                            default="z-score",
                            choices=["z-score", "min-max"]
                        )

    try:
        args = parser.parse_args()
        print(args)

        if args.action == "split" and not args.dataset_path:
            raise ValueError("You must provide a dataset path when splitting the data.")

        if args.action == "split":
            split_data(args.dataset_path, args.ratio, args.random_seed)

    except Exception as e:
        print(f"Error: {e}")