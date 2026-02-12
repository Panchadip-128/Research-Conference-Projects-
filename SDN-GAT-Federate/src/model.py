import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATv2Classifier(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GATv2Classifier, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_channels, heads=2, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin(x)
        return x
