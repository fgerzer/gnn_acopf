from torch.nn import functional as F
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init


def build_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
    return optimizer, lr_scheduler


class GeomOPFLoss:
    def __init__(self, function, vals_to_summarize=None):
        self.function = function
        self.vals_to_summarize = vals_to_summarize

    def __call__(self, input, target, **kwargs):
        total = 0
        relevant_keys = self.vals_to_summarize if self.vals_to_summarize is not None else target.keys()
        for v in relevant_keys:
            relevant_indices = ~torch.isnan(target[v])
            if not torch.all(relevant_indices):
                total = total + self.function(input[v][relevant_indices], target[v][relevant_indices])
            else:
                total = total + self.function(input[v], target[v])
        return total / len(relevant_keys)


class PowerBaseModel(nn.Module):
    optimizer = None
    lr_scheduler = None

    def epoch_finished(self, epoch_number=None):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def update_step(self, data, target):
        self.optimizer.zero_grad()
        output, loss = self.compute_loss(data, target)
        loss.backward()
        self.optimizer.step()
        return output, loss

    def build_optimizer_scheduler(self):
        self.optimizer, self.lr_scheduler = build_optimizer(self)

    def compute_loss(self, data, target):
        output = self(data)
        loss = GeomOPFLoss(F.mse_loss)(output, target)
        return output, loss




class GraphAwareBatchNorm(torch.nn.Module):
    """
    This is a simple reimplementation of the BatchNorm which takes into account that batch statistics are
    in fact *wrong* when using PyTorch-geometric's convention of the first dimension being all the nodes from all the
    batch entries. This computes batch statistics correctly, i.e. it divides mean and variance not by total number
    of nodes but by the batchsize.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, use_aggregate_statistics=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.reset_parameters()
        self.use_aggregate_statistics = use_aggregate_statistics

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.statistics_computed = False
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x, batch=None):
        self._check_input_dim(x)
        batch_mean = (torch.sum(x, dim=0) / x.shape[0])
        batch_var = (torch.sum((x - batch_mean) ** 2, dim=0) / x.shape[0])
        if self.training:
            self.num_batches_tracked += 1
            mean = batch_mean
            variance = batch_var
            self.running_mean = self.running_mean * (1 - self.momentum) + self.momentum * batch_mean.detach()
            self.running_var = self.running_var * (1 - self.momentum) + self.momentum * batch_var.detach()
        else:
            # TODO: For some reason, the aggregate statistics do not work correctly.

            if self.use_aggregate_statistics:
                mean = self.running_mean
                variance = self.running_var
            else:
                mean = batch_mean
                variance = batch_var
            # print(torch.norm(self.running_mean - batch_mean))
            # print(torch.norm(self.running_var - batch_var))
        # normalized = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
        normalized = (x - mean) / (variance + self.eps) ** 0.5
        scaled = self.weight * normalized + self.bias
        return scaled

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
