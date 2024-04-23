import flwr as fl
import torch

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 num_class) -> None:
        super().__init__()

        self.trainload = trainloader
        self.valloader = valloader

        self.model = Net(num_classes) # Initialisation model

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k,v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Copy the parameters sent by the server into the client's local model
        self.set_parameters(parameters)
