import torch.nn as nn


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2*256,
        hidden_dim: int = 2*256,
        output_dim: int = 1,
        num_layers: int = 3,
        activation: nn.Module = nn.ReLU,
        output_activation: nn.Module = nn.Sigmoid(),
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.output_activation = output_activation
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
