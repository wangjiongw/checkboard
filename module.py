from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class HanLayer(nn.Module):
    def __init__(self, in_channel, bias=True):
        super().__init__()
        self.in_channel = in_channel
        self.bias = bias
        self.u = nn.Embedding(1, self.in_channel)
        if self.bias:
            self.b = nn.Embedding(1, self.in_channel)

    def forward(self, x):
        '''
        x: bs * C_in
        return: bs * C_out
        function:
                z = x - (2 * (u^T * x) / ||u||^2) * u + b
                y = abs(z)
        '''
        # pdb.set_trace()
        norm = torch.norm(self.u.weight, p=2)
        u = self.u.weight.squeeze(0).repeat(x.shape[0], 1)
        b = self.b.weight.squeeze(0).repeat(x.shape[0], 1)
        x = x - 2. * torch.matmul(u.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1) / norm * u + b
        x = torch.abs(x)
        return x

class FCMethod(nn.Module):
    def __init__(self, depth=6, width=100, layer_type='fc_layer', act_type='relu'):
        super().__init__()
        self.width = width
        self.depth = depth
        if layer_type == 'fc_layer':
            layer_class = nn.Linear
        elif layer_type == 'han_layer':
            layer_class = HanLayer
        if act_type == 'relu':
            self.act = F.relu
        elif act_type == 'abs':
            self.act = torch.abs
        self.fc_start = nn.Sequential(
            nn.Linear(2, self.width),
            # nn.ReLU()
        )
        self.fc_last = nn.Sequential(
            nn.Linear(self.width, 2)
        )
        self.fc_mid = nn.ModuleList()
        for iLayer in range(self.depth):
            self.fc_mid.append(layer_class(self.width, self.width))
            # self.fc_mid.append(nn.ReLU())
        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)
            # nn.init.constant_(m.weight, 0)

    def forward(self, x):
        x = self.act(self.fc_start(x))
        for Layer in self.fc_mid:
            x = self.act(Layer(x))
        x = self.fc_last(x)

        return x


class MyMethod(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.dropout = 0.0
        self.width = 30
        self.depth = 20
        self.fc_start = nn.Sequential(
            nn.Linear(2, self.width),
            nn.ReLU()
        )
        self.fc_last = nn.Sequential(
            nn.Linear(self.width, 2)
        )
        self.fc_mid = nn.ModuleList()
        for iLayer in range(self.depth):
            self.fc_mid.append(HanLayer(self.width))
        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)
            # nn.init.constant_(m.weight, 0)


    def forward(self, x):
        # pdb.set_trace()
        x = self.fc_start(x)
        for Layer in self.fc_mid:
            x = Layer(x)
        x = self.fc_last(x)
        return x


class MyModelList(nn.Module):
    def __init__(self, model_size, layer='fclayer', act_func='relu'):
        super().__init__()
        # self.models = nn.ModuleList()
        self.model_list = list()
        for size in model_size:
            depth, width = int(size.split('-')[0]), int(size.split('-')[1])
            self.model_list.append(FCMethod(depth, width, layer, act_func))

    def forward(self, x):
        assert len(self.model_list) > 0, "No models found"
        out = self.model_list[0](x)
        for i in range(1, len(self.model_list)):
            out += self.model_list[i](x)
        out = out / float(len(self.model_list))
        return out

    def to_cuda(self):
        for i in range(len(self.model_list)):
            self.model_list[i] = self.model_list[i].cuda()

    def set_train(self):
        for i in range(len(self.model_list)):
            self.model_list[i].train()

    def set_eval(self):
        for i in range(len(self.model_list)):
            self.model_list[i].eval()
