import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    # Parse the config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":

    main()
