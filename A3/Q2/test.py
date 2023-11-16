from encoder import *
import torch

edge_encoder = EdgeEncoder(5)
edge_attr = torch.tensor([1, 1, 1], dtype=torch.long)
edge_attr = edge_attr.reshape(1, -1) #the proper shape for a single sample

edge_encoding = edge_encoder(edge_attr)

print(edge_encoding)

# t1 = torch.tensor([6., 0., 3., 6., 0., 0., 1., 0., 0.])
# t2 = torch.tensor([ 1.0237,  1.5696, -0.0437,  0.7301,  0.8651,  0.2594])

# t1 = torch.cat((t1, t2))

# print(t1)

node_encoder = NodeEncoder(5)
node_features = torch.tensor([5, 0, 4, 5, 3, 0, 2, 0, 0], dtype=torch.long)
node_features = node_features.reshape(1, -1)
node_encoding = node_encoder(node_features)

print(node_encoding)