import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.data import Data


class EquivariantMPLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, act: nn.Module):
        super(EquivariantMPLayer, self).__init__()
        self.act = act
        self.residual_proj = nn.Linear(in_channels, hidden_channels, bias=False)

        message_input_size = 2 * in_channels + 1

        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_size, hidden_channels),
            act,
        )

        self.node_update_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            act,
        )

    def node_message_function(
        self,
        source_node_embed: torch.Tensor,
        target_node_embed: torch.Tensor,
        node_dist: torch.Tensor,
    ) -> torch.Tensor:
        message_repr = torch.cat(
            (source_node_embed, target_node_embed, node_dist), dim=-1
        )
        return self.message_mlp(message_repr)

    def compute_distances(
        self, node_pos: torch.Tensor, edge_index: torch.LongTensor
    ) -> torch.Tensor:
        row, col = edge_index
        xi, xj = node_pos[row], node_pos[col]

        rsdist = (xi - xj).pow(2).sum(dim=1, keepdim=True)
        return rsdist

    def forward(
        self,
        node_embed: torch.Tensor,
        node_pos: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> torch.Tensor:
        row, col = edge_index
        dist = self.compute_distances(node_pos, edge_index)

        node_messages = self.node_message_function(
            node_embed[row], node_embed[col], dist
        )

        aggr_node_messages = scatter(
            node_messages, col, dim=0, dim_size=node_embed.size(0), reduce="sum"
        )

        new_node_embed = self.residual_proj(node_embed) + self.node_update_mlp(
            torch.cat((node_embed, aggr_node_messages), dim=-1)
        )

        return new_node_embed


class CLASPEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        final_embedding_size: int = None,
        target_size: int = 1,
        num_mp_layers: int = 2,
        act: nn.Module = nn.ReLU(),
    ):
        super(CLASPEncoder, self).__init__()
        if final_embedding_size is None:
            final_embedding_size = hidden_channels

        self.act = act
        self.f_initial_embed = nn.Linear(in_channels, hidden_channels)

        self.message_passing_layers = nn.ModuleList()
        channels = [hidden_channels] * (num_mp_layers) + [final_embedding_size]
        for d_in, d_out in zip(channels[:-1], channels[1:]):
            layer = EquivariantMPLayer(d_in, d_out, self.act)
            self.message_passing_layers.append(layer)

        self.aggregation = SumAggregation()
        self.f_predict = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            self.act,
            nn.Linear(final_embedding_size, target_size),
        )

    def encode(self, data: Data) -> torch.Tensor:
        node_embed = self.f_initial_embed(data.x)

        for mp_layer in self.message_passing_layers:
            node_embed = mp_layer(node_embed, data.pos, data.edge_index)
        return node_embed

    def _predict(
        self, node_embed: torch.Tensor, batch_index: torch.Tensor
    ) -> torch.Tensor:
        aggr = self.aggregation(node_embed, batch_index)
        return self.f_predict(aggr)

    def forward(self, data: Data) -> torch.Tensor:
        node_embed = self.encode(data)
        pred = self._predict(node_embed, data.batch)
        return pred
