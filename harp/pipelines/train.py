from .. import train_dac, train_harp


def main(model=None, config_path="harp/configs/train_harp.yaml"):
    if model == "harp":
        train_harp.main(config_path=config_path)
    elif model == "dac":
        train_dac.main(config_path=config_path)
    else:
        raise ValueError(f"Unknown model: {model}. Choose from: harp, dac")
