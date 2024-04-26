import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    # 1. Parse the config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare the dataset
    trainloaders, validationloaders, testloaders = prepare_dataset(cfg.num_clients,
                                                                   cfg.batch_size)
    print(len(trainloaders), len(trainloaders[0].dataset))

    # 3. Define your clients
    


if __name__ == "__main__":

    main()
