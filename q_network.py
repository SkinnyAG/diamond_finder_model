import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, observation_size, action_size):
        super(QNetwork, self).__init__()
        """
        coordinates_size = observation_space[0].nvec.size
        surrounding_blocks_size = len(observation_space[1].nvec)
        tilt_size = observation_space[2].n

        input_size = coordinates_size + surrounding_blocks_size + tilt_size
        """

        self.fc1 = nn.Linear(observation_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        coordinates = x[:, :3].float()
        surrounding_blocks = x[:, 3:13].float()
        tilt = x[:,13:].float()

        x = torch.cat([coordinates, surrounding_blocks, tilt], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values