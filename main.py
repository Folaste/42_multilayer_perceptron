import argparse

from src.parser_utils import ratio_type, positive_int
from src.split_data import split_data
from src.neural_network import deep_neural_network
from src.predict import final_prediction


def parse_args():

    description = """
        Multilayer Perceptron is a program that can be used to train a neural network.
        You can use it to split the data, train the network, and predict the results.
    """
    parser = argparse.ArgumentParser(description=description)

    mandatory_parser = parser.add_argument_group(title="Mandatory arguments")

    mandatory_parser.add_argument("-a", "--action",
                            type=str,
                            required=True,
                            help="Action to perform. Possible values: split, train, predict",
                            choices=["split", "train", "predict"]
                        )

    mandatory_parser.add_argument("-s", "--random_seed",
                            type=int,
                            required=False,
                            default=None,
                            help="Random seed to use for shuffling data in splitting program "
                                 "and initialize weights and biases. (default = None)."
                        )

    split_parser = parser.add_argument_group(title="Split arguments")

    split_parser.add_argument("-d", "--dataset_path",
                            type=str,
                            required=False,
                            help="Path to the input dataset."
                        )

    split_parser.add_argument("-dr", "--dev_ratio",
                            type=ratio_type,
                            required=False,
                            help="Ratio of the validation set to the total dataset (between 0 and 1, default = 0.15).",
                            default=0.15
                        )

    split_parser.add_argument("-tr", "--test_ratio",
                          type=ratio_type,
                          required=False,
                          help="Ratio of the validation set to the total dataset (between 0 and 1, default = 0.15).",
                          default=0.15
                      )

    split_parser.add_argument("-n", "--normalisation_method",
                            type=str,
                            required=False,
                            help="Normalisation method to use for training dataset (default = z-score).",
                            default="z-score",
                            choices=["z-score", "min-max"]
                        )


    train_parser = parser.add_argument_group(title="Train arguments")

    train_parser.add_argument("-e", "--epochs",
                            type=positive_int,
                            required=False,
                            help="Number of epochs to train the network (default = 100).",
                            default=100
                        )

    train_parser.add_argument("-lr", "--learning_rate",
                            type=float,
                            required=False,
                            help="Learning rate to use for training program (default = 0.0314).",
                            default=0.0314
                        )

    train_parser.add_argument("-b", "--batch_size",
                            type=positive_int,
                            required=False,
                            help="Batch size to use for training program (default = 8).",
                            default=8
                        )

    train_parser.add_argument("-l", "--layers",
                            nargs='+',
                            required=False,
                            type=positive_int,
                            help="Number of neurons in each layer to use for training program (default = [32, 16, 8]).",
                            default=[32, 16, 8]
                        )
    train_parser.add_argument("-af", "--activation_function",
                            type=str,
                            choices=["relu", "sigmoid", "tanh"],
                            required=False,
                            help="Activation function to use for training program (default = sigmoid).",
                            default="sigmoid"
                        )

    train_parser.add_argument("-w", "--weights_initializers",
                            type=str,
                            choices=["heNormal", "heUniform"],
                            required=False,
                            help="Weights initializers to use for training program (default = heNormal).",
                            default="heNormal"
                        )

    predict_parser = parser.add_argument_group(title="Predict arguments")

    predict_parser.add_argument("-m", "--model_path",
                            type=str,
                            required=False,
                            help="Path to the model to use for prediction.")

    predict_parser.add_argument("-v", "--verbose",
                                    action="store_true",
                                    help="Print each sample prediction results."
                                )

    predict_parser.add_argument("-cm", "--confusion_matrix",
                                action="store_true",
                                help="Print the confusion matrix."
                                )

    return parser.parse_args()


def main():
    try:
        args = parse_args()

        match args.action:
            case "split":
                if not args.dataset_path:
                    raise ValueError("You must provide a dataset path when splitting the data.")
                if (args.test_ratio + args.dev_ratio) >= 1:
                    raise ValueError("test_ratio + dev_ratio must be lower than 1.")
                split_data(args.dataset_path, args.dev_ratio, args.test_ratio, args.random_seed,
                           args.normalisation_method)

            case "train":
                if len(args.layers) < 2:
                    raise ValueError("You must provide at least 2 hidden layers when training.")
                deep_neural_network("data/X_training_data.csv", "data/y_onehot_training_data.csv",
                                    "data/X_dev_data.csv", "data/y_onehot_dev_data.csv", args.layers,
                                    args.learning_rate, args.epochs, args.batch_size, args.random_seed)

            case "predict":
                if not args.model_path:
                    raise ValueError("You must provide a model path when predicting.")
                final_prediction(args.model_path, "data/X_test_data.csv", "data/y_onehot_test_data.csv", args.verbose,
                                 args.confusion_matrix)

            case _:
                print("Action not recognized.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()