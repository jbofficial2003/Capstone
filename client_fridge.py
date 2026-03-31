import flwr as fl
import torch
from model import SharedModel
from utils import load_fridge_data

X, y = load_fridge_data("data/fridge.csv")

model = SharedModel(X.shape[1])

class FridgeClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(2):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        return self.get_parameters(config), len(X), {}

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)
        loss_fn = torch.nn.CrossEntropyLoss()

        output = model(X)
        loss = loss_fn(output, y)

        return float(loss.detach()), len(X), {}

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FridgeClient()
)