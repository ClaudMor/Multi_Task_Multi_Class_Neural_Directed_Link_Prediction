from torch_geometric.utils import to_edge_index,negative_sampling

train_edge_index = to_edge_index(train_data_directional.edge_index)[0].cpu()
num_nodes = train_data_directional.x.shape[0]
all_destinations = torch.arange(num_nodes, dtype = torch.float)
src =  train_edge_index[:, train_edge_index[0, :] == 0][1]

(all_destinations.unsqueeze(1) != src).all(dim = 1)

neg_destinations = torch.tensor([torch.tensor(all_destinations[(all_destinations.unsqueeze(1) != train_edge_index[:, train_edge_index[0, :] == i][1]).any(dim = 1)]).multinomial(100, replacement=False).tolist()  for i in train_edge_index[0,:].unique()])

neg_edge_idx = train_edge_index[0, :].reshape(-1,1).repeat()
neg_edge_idx= torch.stack((train_edge_index[0, :].unique().reshape(-1,1).expand(-1,100), neg_destinations), dim=1)

all_destinations in train_edge_index[:, train_edge_index[0, :] == 0][1]

[all_destinations[el in train_edge_index[:, train_edge_index[0, :] == i][1] for el in all_destinations]]
[all_destinations[el in train_edge_index[:, train_edge_index[0, :] == i][1] for el in all_destinations] for i in train_edge_index[0,:]] #all_destinations[all_destinations]
torch.arange(num_nodes, dtype = torch.float)[].multinomial(100, replacement=False)

neg_samples = torch.tensor([negative_sampling(train_edge_index[:, train_edge_index[0, :] == i], num_nodes = num_nodes, num_neg_samples = 100).tolist() for i in train_edge_index[0,:]])


a = torch.arange(10)
b = torch.arange(2, 7)[torch.randperm(5)]