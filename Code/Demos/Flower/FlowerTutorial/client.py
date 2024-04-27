from typing import Dict
import flwr as fl
from flwr.common import NDArrays
import torch

from model import Net, train, test

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

    # Extracts list from model
    def get_parameters(self, config: Dict[str, Scalar]):
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def fit(self, parameters, config):
        # Copy the parameters sent by the server into the client's local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Do local training
        train(self.model, self.trainload, optim, epochs, self.device) # Model no longer has weights sent my the server

        return self.get_parameters(), len(self.trainload), {}
