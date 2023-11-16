import math
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader

from encoder import *

DATA_DIR = "../dataset/dataset_2/train/"
CLASS_OR_REGR = "class"
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Initialize the NodeEncoder
NUM_NDIMS = 5
node_encoder = NodeEncoder(NUM_NDIMS)

# Initialize the EdgeEncoder
NUM_EDIMS = 5
edge_encoder = EdgeEncoder(NUM_EDIMS)

# Read the csv files into dataframes
edges_df = pd.read_csv(DATA_DIR + 'edges.csv.gz', compression='gzip', header=None)
num_nodes_df = pd.read_csv(DATA_DIR + 'num_nodes.csv.gz', compression='gzip', header=None)
num_edges_df = pd.read_csv(DATA_DIR + 'num_edges.csv.gz', compression='gzip', header=None)
graph_labels_df = pd.read_csv(DATA_DIR + 'graph_labels.csv.gz', compression='gzip', header=None)
node_features_df = pd.read_csv(DATA_DIR + 'node_features.csv.gz', compression='gzip', header=None)
edge_features_df = pd.read_csv(DATA_DIR + 'edge_features.csv.gz', compression='gzip', header=None)

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
    
    encoded_edge_features = [torch.empty(3 * NUM_EDIMS) for _ in range(num_edges)]

    # # Initialize a list to hold the combined edge features for each node
    combined_edge_features = [torch.empty((0, edge_features.shape[1])) for _ in range(num_nodes)]

    # Combine edge features for each node
    for idx, edge in enumerate(edges):
        combined_edge_features[edge[0]] = torch.cat((combined_edge_features[edge[0]], edge_features[idx].unsqueeze(0)))
        combined_edge_features[edge[1]] = torch.cat((combined_edge_features[edge[1]], edge_features[idx].unsqueeze(0)))

        encoded_edge_features[idx] = edge_encoder(edge_features[idx].long().reshape(1, -1)).view(-1).detach()

    encoded_node_features = [torch.empty(9 * NUM_NDIMS + 3 * NUM_EDIMS) for _ in range(num_nodes)]
    # Encode the combined edge features and append to node features
    for node in range(num_nodes):
        edge_encoding = edge_encoder(combined_edge_features[node].long())
        node_encoding = node_encoder(node_features[node].long().reshape(1, -1)).view(-1)
        encoded_node_features[node] = torch.cat((node_encoding, edge_encoding.mean(dim = 0))).detach()

    data = Data(x=encoded_node_features, edge_index=edges.t().contiguous(), edge_attr=encoded_edge_features)
    if CLASS_OR_REGR == "class":
        data.y = torch.tensor([int(graph_labels_df.iloc[i, 0])], dtype=torch.long)  # Set the label
    else:
        data.y = torch.tensor([graph_labels_df.iloc[i, 0]], dtype=torch.double)  # Set the label
    data_list.append(data)

print("Data Loaded")
exit()

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
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # self.pool = torch.nn.AdaptiveAvgPool1d(1)  # Pooling layer to aggregate node features
        # self.fc = torch.nn.Linear(hidden_channels, out_channels)

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data) -> torch.Tensor:
        print(data)
        data.x = self.conv1(data.x, data.edge_index)
        print(data)
        data.x = F.relu(data.x)
        data.x = self.conv2(data.x, data.edge_index)
        print(data)
        data.x = F.relu(data.x)
        x = global_add_pool(data.x, data.batch)

        # Final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = torch.sigmoid(x)

        # x = (x[:, 0] > x[:, 1]).long()

        # x = F.log_softmax(x, dim=1)

        return x


def custom_collate(batch):
    batch = Batch.from_data_list(batch)
    graph_num_nodes = [graph.num_nodes for graph in batch]
    batch.batch = torch.cat([torch.full((num_nodes,), graph_idx) for graph_idx, num_nodes in enumerate(graph_num_nodes)])
    return batch

dataset = CustomDataset(data_list)
dataloader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

'''
DataLoader is a collection of batches. Each batch is a Batch object, which is a collection of <BATCH_SIZE> graphs as follows:

for data in dataloader:
    graph_idx = 0  # The index of the graph you want to access: 0 to BATCH_SIZE-1
    subgraph = data[graph_idx]

'''

class_model = GCN(in_channels=dataset.num_features, hidden_channels=32, out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=5e-4)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(NUM_EPOCHS):
    for batch in dataloader:
        output = class_model(batch)
        # print(output)
        # output = torch.argmax(output, dim=1)
        gold = F.one_hot(batch.y, num_classes=2).to(dtype=torch.float)
        loss = criterion(output, gold)
        print(loss)
        # target = batch.y.float()
        # loss = criterion(output, target.float)
        # y_one_hot = F.one_hot(batch.y.float(), 2)
        # loss = criterion(output, y_one_hot)
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