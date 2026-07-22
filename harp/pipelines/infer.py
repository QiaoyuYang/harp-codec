from .. import infer_dac, infer_harp


def main(model=None, args=None, config_path=None):
    if model == "harp":
        infer_harp.main(args=args, config_path=config_path)
    elif model == "dac":
        infer_dac.main(args=args, config_path=config_path)
    else:
        raise ValueError(f"Unknown model: {model}. Choose from: harp, dac")
