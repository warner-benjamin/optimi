import pytest
import torch
import optimi
import inspect
from optimi.optimizer import OptimiOptimizer

# Dynamically collect all optimizer class names in the optimi module
OPTIMIZERS = sorted([
    name for name, cls in inspect.getmembers(optimi)
    if inspect.isclass(cls)
    and issubclass(cls, OptimiOptimizer)
])

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, device, dtype):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.act = torch.nn.Mish()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.fc2 = torch.nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.fc2(self.norm(self.act(self.fc1(x))))


@pytest.mark.cpu
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("additional_layer", [None, 'fc2'])
def test_param_groups_weight_decay(optimizer, additional_layer):
    model = MLP(10, 20, torch.device('cpu'), torch.float16)

    additional_layers = [additional_layer] if additional_layer is not None else None
    params = optimi.param_groups_weight_decay(model, weight_decay=1e-2, additional_layers=additional_layers)

    filtered_wd_pg = False
    wd_pg = False

    # first test that we have two groups, one with weight decay and one without
    for param_group in params:
        if param_group["weight_decay"] != 0:
            assert param_group["weight_decay"] == 1e-2, "Expected weight decay to be 1e-2"
            if additional_layer is not None:
                assert len(param_group["params"]) == 1, "Expected only fc1.weight in the group with weight decay"
            else:
                assert len(param_group["params"]) == 2, "Expected fc1.weight & fc2.weight in the group with weight decay"
            wd_pg = True
        if param_group["weight_decay"] == 0:
            if additional_layer is not None:
                assert len(param_group["params"]) == 4, "Expected fc1.bias, norm.weight, norm.biasm & fc2.weight in the group without weight decay"
            else:
                assert len(param_group["params"]) == 3, "Expected fc1.bias, norm.weight, & norm.bias in the group without weight decay"
            filtered_wd_pg = True

    assert filtered_wd_pg, "Expected a parameter group without weight decay"
    assert wd_pg, "Expected a parameter group with weight decay"

    # now test that the optimizer also has both groups, one with weight decay and one without
    # weight decay passed to the optimizer will be ignored
    opt = getattr(optimi, optimizer)(params, lr=1e-3, weight_decay=1e-3)

    filtered_wd_pg = False
    wd_pg = False

    for param_group in opt.param_groups:
        if param_group["weight_decay"] != 0:
            assert param_group["weight_decay"] == 1e-2, "Expected weight decay to be 1e-2"
            if additional_layer is not None:
                assert len(param_group["params"]) == 1, "Expected only fc1.weight in the group with weight decay"
            else:
                assert len(param_group["params"]) == 2, "Expected fc1.weight & fc2.weight in the group with weight decay"
            wd_pg = True
        if param_group["weight_decay"] == 0:
            if additional_layer is not None:
                assert len(param_group["params"]) == 4, "Expected fc1.bias, norm.weight, norm.biasm & fc2.weight in the group without weight decay"
            else:
                assert len(param_group["params"]) == 3, "Expected fc1.bias, norm.weight, & norm.bias in the group without weight decay"
            filtered_wd_pg = True

    assert filtered_wd_pg, "Expected an optimizer parameter group without weight decay"
    assert wd_pg, "Expected an optimizer parameter group with weight decay"