import argparse
from pprint import pprint


def main(args):
    if args.model == 'cnn':
        pass

    elif args.model == 'lstm':
        pass

    elif args.model == 'hybrid':
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with given parameters.")
    parser.add_argument('--model_type',
                        choices=['cnn', 'lstm', 'hybrid'],
                        required=True,
                        help='Model choice to be evaluated.')
    parser.add_argument('--model_path',
                        required=True,
                        help='Path where the trained model was saved.')
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Batch size used for training the model.")

    args = parser.parse_args()

    print("Evaluating:")
    pprint(vars(args))
    print("-" * 30)

    main(args)
