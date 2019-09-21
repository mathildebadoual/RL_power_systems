import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):

    # timestep of day
    # soe
    # last raise clearing price
    # prediction of demand
    # clearing energy price
    # last cleared energy price maybeee
    # last week same day clearing raise price
    # yesterday same timestep clearing raise price
    # bids from other people - maybe later
    # 2 artificial time-dependent features.

    def __init__(self):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CriticNet(nn.Module):

    # timestep of day
    # soe
    # last raise clearing price
    # prediction of demand
    # clearing energy price
    # last cleared energy price maybeee
    # last week same day clearing raise price
    # yesterday same timestep clearing raise price
    # bids from other people - maybe later
    # 2 artificial time-dependent features.

    def __init__(self):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# code to create a NN
net = ActorNet()
input = torch.randn(1, 5)
# feed forward
out = net(input)
# get rid of stored gradients
net.zero_grad()
# backpropogates the gradient wrt the output layer
# the argument should be a tensor of ones that matches the size of the output
out.backward(torch.ones(1,3))
