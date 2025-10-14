import torch
import numpy as np

topology = 'run.gro'
initial_paths = ['initial.xtc']

def descriptors_function(traj, verbose=False):
    return np.array([
        frame.positions[0, 0] for frame in
        tqdm(traj, disable=not verbose, position=0)]).reshape(-1, 1)

def states_function(traj, verbose=False):
    return np.array([
        'A' if frame.positions[0, 0] <= 0.0 else
        'B' if frame.positions[0, 0] >= 1.0 else 'R' for frame in
        tqdm(traj, disable=not verbose, position=0)], dtype='<U1')

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_kwargs = {}
        n = 16
        self.input = torch.nn.Linear(1, n)
        self.layer = torch.nn.Linear(n, n)
        self.activation = torch.nn.ReLU(n)
        self.output = torch.nn.Linear(n, 1)
        self.reset_parameters()
    def forward(self, x):
        x = self.activation(self.input(x))
        x = self.activation(self.layer(x))
        x = self.output(x)
        return x
    def reset_parameters(self):
        self.input.reset_parameters()
        self.layer.reset_parameters()
        self.output.reset_parameters()

network = Network()

def process_descriptors(descriptors):
    return descriptors

def values_function(descriptors):
    if not len(descriptors):
        return np.zeros(0)
    
    global network
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    network.eval()
    
    # initialize
    results = []
    descriptors = process_descriptors(descriptors)
    
    # compute in batches
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(
            descriptors, batch_size=4096, shuffle=False):
            batch = batch.to(device=device, dtype=dtype)
            output = network(batch).detach().cpu().numpy().ravel()
            results.append(output)
    
    # return
    return np.concatenate(results)
