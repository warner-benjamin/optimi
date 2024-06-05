import pytest
import torch
import optimi


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
@pytest.mark.parametrize("optimizer", ['Adam', 'Adan', 'Lion', 'RAdam', 'Ranger', 'SGD', 'StableAdamW'])
def test_param_groups_weight_decay(optimizer):
    model = MLP(10, 20, torch.device('cpu'), torch.float16)

    # first test that we have two groups, one with weight decay and one without

    params = optimi.param_groups_weight_decay(model, weight_decay=1e-2)

    filtered_wd_pg = False
    wd_pg = False

    for param_group in params:
        if param_group["weight_decay"] != 0:
            assert param_group["weight_decay"] == 1e-2, "Expected weight decay to be 1e-2"
            assert len(param_group["params"]) == 2, "Expected fc1.weight & fc2.weight in the group with weight decay"
            wd_pg = True
        if param_group["weight_decay"] == 0:
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
            assert len(param_group["params"]) == 2, "Expected fc1.weight & fc2.weight in the group with weight decay"
            wd_pg = True
        if param_group["weight_decay"] == 0:
            assert len(param_group["params"]) == 3, "Expected fc1.bias, norm.weight, & norm.bias in the group without weight decay"
            filtered_wd_pg = True

    assert filtered_wd_pg, "Expected a parameter group without weight decay"
    assert wd_pg, "Expected a parameter group with weight decay"