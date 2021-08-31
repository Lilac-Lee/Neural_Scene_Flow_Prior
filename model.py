import torch


class Neural_Prior(torch.nn.Module):
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu'):
        super().__init__()
        # input layer (default: xyz -> 128)
        self.layer1 = torch.nn.Linear(dim_x, filter_size)
        # hidden layers (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, filter_size)
        self.layer4 = torch.nn.Linear(filter_size, filter_size)
        self.layer5 = torch.nn.Linear(filter_size, filter_size)
        self.layer6 = torch.nn.Linear(filter_size, filter_size)
        self.layer7 = torch.nn.Linear(filter_size, filter_size)
        self.layer8 = torch.nn.Linear(filter_size, filter_size)
        # output layer (default: 128 -> 3)
        self.layer9 = torch.nn.Linear(filter_size, 3)

        # activation functions
        if act_fn == 'relu':
            self.act_fn = torch.nn.functional.relu
        elif act_fn == 'sigmoid':
            self.act_fn = torch.nn.functional.sigmoid

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        x = self.act_fn(self.layer1(x))
        x = self.act_fn(self.layer2(x))
        x = self.act_fn(self.layer3(x))
        x = self.act_fn(self.layer4(x))
        x = self.act_fn(self.layer5(x))
        x = self.act_fn(self.layer6(x))
        x = self.act_fn(self.layer7(x))
        x = self.act_fn(self.layer8(x))
        x = self.layer9(x)

        return x
    