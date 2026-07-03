import argparse


DEFAULT_MODELS = ["duck", "unetpp", "bnet", "unet", "bnet34", "unext", "dga", "pham"]
DEFAULT_DATASETS = ["kvasir", "clinicdb", "isic2018"]


def _as_list(values, defaults):
    if not values:
        return defaults
    result = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="FYP-Net training and testing entry point.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train selected models on selected datasets")
    train_parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    train_parser.add_argument("--model", nargs="*", help="Model names, e.g. bnet unet or bnet,unet")
    train_parser.add_argument("--dataset", nargs="*", help="Dataset names, e.g. isic2018 kvasir")
    train_parser.add_argument("--analyze-only", action="store_true", help="Only print FLOPs and parameters")

    test_parser = subparsers.add_parser("test", help="Test selected models on selected datasets")
    test_parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    test_parser.add_argument("--model", nargs="*", help="Model names, e.g. bnet34")
    test_parser.add_argument("--dataset", nargs="*", help="Dataset names, e.g. isic2018 clinicdb kvasir")
    test_parser.add_argument("--repeat", type=int, default=1, help="Repeat times for random-seed testing")

    subparsers.add_parser("list", help="List built-in model and dataset names")
    return parser


def run_train(args):
    from train_model import Trainer

    for model_name in _as_list(args.model, DEFAULT_MODELS):
        for dataset_name in _as_list(args.dataset, DEFAULT_DATASETS):
            print("-----------------")
            print(model_name + " || " + dataset_name)
            trainer = Trainer(args.config, model_name=model_name, dataset_name=dataset_name)
            if args.analyze_only:
                trainer.analyze((3, 224, 224))
            else:
                trainer.train()


def run_test(args):
    import random
    from test_model import Tester, set_random_seed

    repeat_times = max(args.repeat, 1)
    seeds = [random.randint(1, 10000) for _ in range(repeat_times)]
    print(seeds)

    for model_name in _as_list(args.model, ["bnet34"]):
        for dataset_name in _as_list(args.dataset, DEFAULT_DATASETS):
            print("------------------------------------------")
            print(model_name + " || " + dataset_name)
            tester = Tester(args.config, _model_name=model_name, _dataset_name=dataset_name)
            dice_avg, miou_avg, acc_avg, prec_avg, recall_avg = 0, 0, 0, 0, 0

            for run in range(repeat_times):
                seed = seeds[run]
                set_random_seed(seed)
                print(f"Running {model_name} on {dataset_name}, Seed: {seed}")
                a, b, c, d, e = tester.test()

                dice_avg += a
                miou_avg += b
                acc_avg += c
                prec_avg += d
                recall_avg += e

            print(f"Average Results for {model_name} on {dataset_name}:")
            print(f"Dice Avg: {dice_avg / repeat_times:.6f}")
            print(f"Miou Avg: {miou_avg / repeat_times:.6f}")
            print(f"Accuracy Avg: {acc_avg / repeat_times:.6f}")
            print(f"Precision Avg: {prec_avg / repeat_times:.6f}")
            print(f"Recall Avg: {recall_avg / repeat_times:.6f}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        print("Models: " + ", ".join(DEFAULT_MODELS))
        print("Datasets: " + ", ".join(DEFAULT_DATASETS))
    elif args.command == "train":
        run_train(args)
    elif args.command == "test":
        run_test(args)


if __name__ == "__main__":
    main()
