from torch_geometric.datasets import QM9

if __name__=='__main__':
    qm9 = QM9(root='./data/qm9-2.4.0/')
    print(qm9)
