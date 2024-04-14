import torch.nn as nn
from irl.utils import *


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh', log_std=None):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        if log_std is not None:
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        else:
            self.action_log_std = nn.Linear(last_dim, 1)
            self.action_log_std.weight.data.mul_(0.1)
            self.action_log_std.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        if not isinstance(self.action_log_std, nn.modules.linear.Linear):         # if log_std was not None
            action_log_std = self.action_log_std.expand_as(action_mean)
        else:                                                           # if trainable log_std
            action_log_std = self.action_log_std(x)
            action_log_std = torch.sigmoid(action_log_std) * -2.30      # should be between -0.0 and -2.30
            action_log_std = action_log_std.expand_as(action_mean)      # make two out of one
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        print("action_mean", action_mean)
        print("action_log_std", action_log_std)
        print("action_std",action_std)
        # x.shape (bx12, 16)
        # action_mean.shpae (bx12, 2)
        # print("action_mean", action_mean.shape)
        # print("\n action_mean: \n shape:", action_mean.shape, "\n values: ", action_mean)
        # print("\n action_mean: \n shape:", action_mean.shape, "\n values: ", action_mean)
        # print("\n action_log_std: \n shape:", action_log_std.shape, "\n values: ", action_log_std)
        # print("\n action_std: \n shape:", action_std.shape, "\n values: ", action_std)
        # print("\n normal_log_density(actions, action_mean, action_log_std, action_std): \n shape:", normal_log_density(actions, action_mean, action_log_std, action_std).shape, "\n values: ", normal_log_density(actions, action_mean, action_log_std, action_std))
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU(inplace=False)
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        # last_dim = last_dim * 2                 # added for shape fix
        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        # x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))   # added for shape fix
        preprob = self.logic(x)
        prob = torch.sigmoid(preprob)
        return prob # [b,1]


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        # self.batch_norm = nn.BatchNorm1d(state_dim, affine=True)

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        # last_dim = last_dim * 2  # added for shape fix
        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        # x = self.batch_norm(x)
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value