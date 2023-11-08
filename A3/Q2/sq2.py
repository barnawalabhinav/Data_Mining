import math
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader


DATA_DIR = "../dataset/dataset_2/train/"
CLASS_OR_REGR = "class"
BATCH_SIZE = 32
NUM_EPOCHS = 10


# Read the csv files into dataframes
edges_df = pd.read_csv(DATA_DIR + 'edges.csv.gz', compression='gzip')
num_nodes_df = pd.read_csv(DATA_DIR + 'num_nodes.csv.gz', compression='gzip')
num_edges_df = pd.read_csv(DATA_DIR + 'num_edges.csv.gz', compression='gzip')
graph_labels_df = pd.read_csv(DATA_DIR + 'graph_labels.csv.gz', compression='gzip')
node_features_df = pd.read_csv(DATA_DIR + 'node_features.csv.gz', compression='gzip')
edge_features_df = pd.read_csv(DATA_DIR + 'edge_features.csv.gz', compression='gzip')

NUM_GRAPHS = len(graph_labels_df)
data_list = []

nodes_done = 0
edges_done = 0
for i in range(NUM_GRAPHS):
    num_nodes = num_nodes_df.iloc[i, 0]
    num_edges = num_edges_df.iloc[i, 0]
    nodes_done += num_nodes
    edges_done += num_edges

    if math.isnan(graph_labels_df.iloc[i, 0]):
        continue

    node_features = torch.tensor(node_features_df.iloc[nodes_done-num_nodes:nodes_done, :].values, dtype=torch.float)
    edges = torch.tensor(edges_df.iloc[edges_done-num_edges:edges_done, :].values)
    edge_features = torch.tensor(edge_features_df.iloc[edges_done-num_edges:edges_done, :].values, dtype=torch.float)

    data = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features)
    if CLASS_OR_REGR == "class":
        data.y = torch.tensor([int(graph_labels_df.iloc[i, 0])], dtype=torch.int)  # Set the label
    else:
        data.y = torch.tensor([graph_labels_df.iloc[i, 0]], dtype=torch.double)  # Set the label
    data_list.append(data)


class CustomDataset(Dataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class Random_Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, length) -> torch.Tensor:
        return torch.randint(0, self.num_classes, (length,))


# class Linear_Regression(torch.nn.Module):


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)  # Pooling layer to aggregate node features
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # x = self.pool(x.transpose(1, 2)).squeeze(2)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def custom_collate(batch):
    batch = Batch.from_data_list(batch)
    graph_num_nodes = [graph.num_nodes for graph in batch]
    batch.batch = torch.cat([torch.full((num_nodes,), graph_idx) for graph_idx, num_nodes in enumerate(graph_num_nodes)])
    return batch

'''
DataLoader is a collection of batches. Each batch is a Batch object, which is a collection of <BATCH_SIZE> graphs as follows:

for data in dataloader:
    graph_idx = 0  # The index of the graph you want to access: 0 to BATCH_SIZE-1
    subgraph = data[graph_idx]

'''

dataset = CustomDataset(data_list)
dataloader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_model = GCN(in_channels=dataset.num_features, hidden_channels=32, out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(NUM_EPOCHS):
    for data in dataloader:
        output = class_model(data.x, data.edge_index)
        loss = F.cross_entropy(output, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# random_model = Random_Classifier(num_classes=dataset.num_classes)

# correct_output = 0 
# for data in dataloader:
#     predicted_output = random_model.forward(data.x, length=len(data.y))
#     correct_output += torch.sum(data.y == predicted_output).item()

# print(f"Random Accuracy: {correct_output / NUM_GRAPHS * 100 :.2f}")


# torch.save(class_model.state_dict(), "q2_class_model.pt")