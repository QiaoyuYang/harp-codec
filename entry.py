import argparse
from harp.pipelines import train, infer


def main():
    parser = argparse.ArgumentParser(description='Entry point for HARP')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Run the training pipeline')
    parser.add_argument('-c', '--config', type=str, default='harp/configs/train_harp.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--model', type=str,
                        choices=['harp', 'dac'],
                        default='harp',
                        help='Choose the model to train or infer (default: harp)')
    parser.add_argument('-i', '--infer', action='store_true',
                        help='Run the inference pipeline')

    args, remaining = parser.parse_known_args()

    if args.train:
        print(f"Running training pipeline for {args.model}")
        train.main(model=args.model, config_path=args.config)
    elif args.infer:
        print(f"Running inference pipeline for {args.model}")
        infer.main(model=args.model, args=remaining, config_path=args.config)
    else:
        print("Please provide a valid argument: -t for training or -i for inference")


if __name__ == "__main__":
    main()
