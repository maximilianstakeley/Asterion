
from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, dimension, max_length=2048):
        super().__init__()

        position = torch.arange(
            max_length
        ).unsqueeze(1)

        divisor = torch.exp(
            torch.arange(0, dimension, 2)
            * (-torch.log(torch.tensor(10000.0)) / dimension)
        )

        encoding = torch.zeros(
            max_length, dimension
        )

        encoding[:, 0::2] = torch.sin(
            position * divisor
        )

        encoding[:, 1::2] = torch.cos(
            position * divisor
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0)
        )

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class AsterionTemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dimension=10,
        hidden_dimension=64,
        attention_heads=4,
        layers=3,
        dropout=0.1
    ):
        super().__init__()

        if hidden_dimension % attention_heads:
            raise ValueError(
                "Hidden dimension must divide attention heads"
            )

        self.input_projection = nn.Linear(
            input_dimension,
            hidden_dimension
        )

        self.position = PositionalEncoding(
            hidden_dimension
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=attention_heads,
            dim_feedforward=hidden_dimension * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.normalization = nn.LayerNorm(
            hidden_dimension
        )

        self.state_head = nn.Linear(
            hidden_dimension,
            hidden_dimension
        )

        self.prediction_head = nn.Linear(
            hidden_dimension,
            input_dimension
        )

    def forward(self, sequence, padding_mask=None):
        x = self.input_projection(sequence)

        x = self.position(x)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.normalization(x)

        representation = self.state_head(
            x[:, -1]
        )

        prediction = self.prediction_head(
            representation
        )

        return {
            "representation": representation,
            "next_state_prediction": prediction
        }


def training_step(model, optimizer, sequence, target):
    model.train()

    optimizer.zero_grad()

    output = model(sequence)

    loss = nn.functional.smooth_l1_loss(
        output["next_state_prediction"],
        target
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    return float(loss.detach().cpu())


@torch.no_grad()
def predict(model, sequence):
    model.eval()

    output = model(sequence)

    return {
        "representation": output[
            "representation"
        ].cpu().numpy(),

        "next_state_prediction": output[
            "next_state_prediction"
        ].cpu().numpy()
    }
