import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.utils import scatter
from torchmetrics import (
    R2Score,
    MeanSquaredError,
    MeanAbsoluteError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
    MetricCollection,
)
import pandas as pd
import os


class DirectedMessagePassing(nn.Module):
    """D-MPNN encoder whose hidden states live on directed edges."""

    def __init__(self, d_emb, depth, dropout=0.2):
        super().__init__()
        if depth < 1:
            raise ValueError("D-MPNN depth must be at least 1")

        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # h^0_(v->w) = ReLU(W_i [x_v || e_vw])
        self.edge_input = nn.Linear(2 * d_emb, d_emb, bias=False)
        self.edge_input_norm = nn.BatchNorm1d(d_emb)
        self.message_layers = nn.ModuleList(
            nn.Linear(d_emb, d_emb, bias=False) for _ in range(depth - 1)
        )
        self.message_norms = nn.ModuleList(
            nn.BatchNorm1d(d_emb) for _ in range(depth - 1)
        )

        # h_v = ReLU(W_o [x_v || sum_(u->v) h_(u->v)])
        self.node_output = nn.Sequential(
            nn.Linear(2 * d_emb, d_emb),
            nn.BatchNorm1d(d_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _reverse_edge_index(edge_index, num_nodes):
        """Return the index of w->v for every v->w edge."""
        src, dst = edge_index
        keys = src * num_nodes + dst
        reverse_keys = dst * num_nodes + src
        sorted_keys, order = torch.sort(keys)
        positions = torch.searchsorted(sorted_keys, reverse_keys)

        valid = positions < sorted_keys.numel()
        safe_positions = positions.clamp(max=sorted_keys.numel() - 1)
        valid = valid & (sorted_keys[safe_positions] == reverse_keys)
        if not bool(valid.all()):
            raise ValueError("Every directed edge must have a reverse edge")
        return order[safe_positions]

    def forward(self, node_features, edge_features, edge_index):
        # edge_index follows PyG convention: source in row 0, target in row 1.
        src, dst = edge_index
        num_nodes = node_features.size(0)
        reverse_edge = self._reverse_edge_index(edge_index, num_nodes)

        initial_message = self.activation(
            self.edge_input_norm(
                self.edge_input(
                    torch.cat([node_features[src], edge_features], dim=-1)
                )
            )
        )
        message = initial_message

        for layer, norm in zip(self.message_layers, self.message_norms):
            # For v->w, aggregate u->v messages and explicitly remove w->v.
            incoming = scatter(
                message,
                dst,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
            directed_message = incoming[src] - message[reverse_edge]
            message = self.activation(
                norm(initial_message + layer(directed_message))
            )
            message = self.dropout(message)

        incoming_to_node = scatter(
            message,
            dst,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        node_hidden = self.node_output(
            torch.cat([node_features, incoming_to_node], dim=-1)
        )
        return node_hidden

class CycGNN(nn.Module):
    """
    与 not_mut_gnn 相同的 D-MPNN 主干，但输出方式改为“成对差值”：
    母结构（parent）与子结构（child）分别经过同一套 GNN 权重编码、
    求和池化得到图级 embedding，两者相减后再送入 out_mlp，
    输出 child 与 parent 的渗透性差值预测。
    """

    def __init__(
            self,
            d_emb=128,
            n_heads=4,
            dropout=0.25,
            num_gnn_layer=2,
    ):
        super().__init__()
        self.num_gnn_layer = num_gnn_layer
        # n_heads is kept in the public API so existing training commands remain valid.
        # D-MPNN uses directed edge states and therefore does not use attention heads.
        self.dmpnn = DirectedMessagePassing(d_emb, num_gnn_layer, dropout)
        self.x_projection = nn.Sequential(
            # 21 residue classes + 3 residue properties + 4 assay methods
            nn.Linear(21 + 3 + 4, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
        )
        self.delta_unimol_projection = nn.Sequential(
            nn.Linear(512, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
        )
        self.monomer_property_projection = nn.Sequential(
            nn.Linear(31, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
        )
        self.node_projection = nn.Sequential(
            nn.Linear(d_emb * 3, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(1, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
        )

    def _pool_graph(self, graph_batch, num_graphs):
        """把单个结构（child 或 parent）的残基图编码并求和池化为图级 embedding。"""
        node_feature = graph_batch['node_feature']
        delta_unimol_feature = graph_batch['delta_unimol_feature']
        monomer_property_feature = graph_batch['monomer_property_feature']
        node_batch = graph_batch['node_batch']

        # edge
        edge_index = graph_batch['edge_index']
        edge_feature = graph_batch['edge_attr']

        _x = self.x_projection(node_feature)
        _delta_unimol = self.delta_unimol_projection(delta_unimol_feature)
        _monomer_property = self.monomer_property_projection(
            monomer_property_feature
        )

        h = self.node_projection(
            torch.cat(
                [_delta_unimol, _monomer_property, _x],
                dim=-1,
            )
        )
        e = self.edge_projection(edge_feature)

        # Directed message passing on edges, followed by edge-to-node readout.
        h = self.dmpnn(h, e, edge_index)

        # Read out every graph independently from the disconnected graph batch.
        pooled = scatter(
            h,
            node_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        return pooled

    def forward(self, batch):
        # batch['child'] / batch['parent'] 是两套独立的残基图 batch，
        # batch['y'] 是每一对 (child, parent) 的渗透性差值标签。
        num_graphs = int(batch['y'].numel())

        child_embedding = self._pool_graph(batch['child'], num_graphs)
        parent_embedding = self._pool_graph(batch['parent'], num_graphs)

        diff_embedding = child_embedding - parent_embedding
        out = self.out_mlp(diff_embedding)

        return out.squeeze(-1)

class QJGNN(pl.LightningModule):
    def __init__(
        self,
        d_emb = 128,
        n_heads = 4,
        dropout = 0.25,
        num_gnn_layer = 2,
        lr=1e-4,
        test_names=("internal_test", "faris_data", "merz_data", "nielsen_data"),
        validation_seeds=tuple(range(10)),
    ):

        super().__init__()  # def __init__(self, d_emb, n_heads, n_structure_layer, d_node, dropout=0.5, queue_size=64, **kwargs):
        self.model = CycGNN(d_emb, n_heads, dropout, num_gnn_layer)
        self.lr = lr

        self.criterion = nn.MSELoss(reduction="mean")

        # 评估指标：R2 / MSE / MAE / Pearson r / Spearman r
        # 注意：这里的指标均是在“渗透性差值”这一预测目标上计算的。
        metrics = MetricCollection(
            {
                "R2": R2Score(),
                "MSE": MeanSquaredError(),
                "MAE": MeanAbsoluteError(),
                "pearson": PearsonCorrCoef(),
                "spearman": SpearmanCorrCoef(),
            }
        )

        self.validation_seeds = tuple(validation_seeds)
        self.valid_metrics = nn.ModuleList(
            [
                metrics.clone(prefix=f"val_seed{seed}/")
                for seed in self.validation_seeds
            ]
        )

        # 每个测试集各自独立的一套指标；
        # dataloader_idx=i 对应 test_names[i]（顺序需与 trainer.test 传入的
        # dataloaders 列表一致），例如默认的 0 -> internal_test, 1 -> faris_data。
        self.test_metric_names = list(test_names)
        self.test_metrics = nn.ModuleList(
            [metrics.clone(prefix=f"{name}/") for name in self.test_metric_names]
        )

        # 保存每个测试集的预测结果，用于导出 excel
        self.test_step_outputs = {}

    def forward(self, batch):
        result = self.model(batch)

        return result

    def training_step(self, batch):
        out = self.forward(batch)
        target = batch['y'].reshape(-1)

        loss = self.criterion(out, target)

        self.log_dict(
            {
                "train/loss": loss,
                # "train/perplexity": torch.exp(loss),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=target.numel(),
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self.forward(batch)
        target = batch['y'].reshape(-1)

        loss = self.criterion(out, target)
        seed = self.validation_seeds[dataloader_idx]
        self.valid_metrics[dataloader_idx].update(out.detach(), target.detach())
        self.log(
            f"val_seed{seed}/loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=target.numel(),
            add_dataloader_idx=False,
        )

        return loss

    def on_validation_epoch_end(self):
        metric_names = ("R2", "MSE", "MAE", "pearson", "spearman")
        values_by_name = {name: [] for name in metric_names}
        for index, seed in enumerate(self.validation_seeds):
            seed_metrics = self.valid_metrics[index].compute()
            self.log_dict(
                seed_metrics,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            for name in metric_names:
                values_by_name[name].append(seed_metrics[f"val_seed{seed}/{name}"])
            self.valid_metrics[index].reset()
        means = {
            f"val/{name}_mean": torch.stack(values).mean()
            for name, values in values_by_name.items()
        }
        self.log_dict(means, prog_bar=True, logger=True, sync_dist=True)
        print(
            f"val/pearson_mean {float(means['val/pearson_mean']):.4f}",
            flush=True,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        out = self.forward(batch)
        target = batch['y'].reshape(-1)

        # 根据 dataloader_idx 选择对应测试集的指标进行更新
        self.test_metrics[dataloader_idx].update(out.detach(), target.detach())

        # 记录预测结果（index=子结构，parent_index=本次对比所用的母结构）
        self.test_step_outputs.setdefault(dataloader_idx, []).extend(
            [
                [int(idx), int(parent_idx), prediction.item(), label.item()]
                for idx, parent_idx, prediction, label in zip(
                    batch['index'], batch['parent_index'], out, target
                )
            ]
        )
        return {
            "index": batch["index"].detach().cpu(),
            "parent_index": batch["parent_index"].detach().cpu(),
            "prediction": out.detach().cpu(),
            "label": target.detach().cpu(),
        }

    def on_test_epoch_end(self):
        os.makedirs('../test/result', exist_ok=True)

        for idx, name in enumerate(self.test_metric_names):
            # 计算并记录该测试集的全部指标
            metrics = self.test_metrics[idx].compute()

            # print(
            #     f"[{name}] "
            #     f"R2 {metrics[f'{name}/R2']:.4f} | "
            # )

            self.log_dict(metrics, prog_bar=True, logger=True, sync_dist=True)
            self.test_metrics[idx].reset()

            # # 导出预测结果
            # rows = self.test_step_outputs.get(idx, [])
            # df = pd.DataFrame(rows, columns=['index', 'parent_index', 'Prediction', 'Label'])
            # df.to_excel(f'../test/result/{name}.xlsx', index=False)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.lr,
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     pct_start=0.2,
        # )

        # return optimizer
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step",
            # },
        }
