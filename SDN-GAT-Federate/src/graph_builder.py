import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from .config import GRAPH_BATCH_SIZE, TRAIN_TEST_SPLIT, RANDOM_SEED, BATCH_SIZE

def create_graph_dataset(df_balanced):
    """
    Converts the DataFrame into a list of PyG Graph Data objects.
    """
    print("🚀 Starting Graph Construction...")
    graph_list = []
    batch_size_graph = GRAPH_BATCH_SIZE
    
    for i in range(0, len(df_balanced) - batch_size_graph, batch_size_graph):
        subset = df_balanced.iloc[i : i+batch_size_graph]
        
        src = torch.tensor(subset['ip.src_host'].values, dtype=torch.long)
        dst = torch.tensor(subset['ip.dst_host'].values, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        
        pkt = torch.tensor(subset['tcp.len'].values, dtype=torch.float).view(-1, 1)
        pkt = (pkt - pkt.mean()) / (pkt.std() + 1e-6)
        
        degree = torch.ones_like(pkt)
        x = torch.cat([degree, pkt], dim=1)
        
        label = 1 if subset['Attack_label'].sum() > (batch_size_graph / 2) else 0
        y = torch.tensor([label], dtype=torch.float)
        
        graph_list.append(Data(x=x, edge_index=edge_index, y=y))

    print(f"✅ Graph Conversion Complete: {len(graph_list)} Graphs Created.")
    return graph_list

def get_data_loaders(graph_list):
    """
    Splits the graph dataset and returns train/test loaders.
    """
    train_dataset, test_dataset = train_test_split(graph_list, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader
