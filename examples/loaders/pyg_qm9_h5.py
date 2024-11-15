import os
import h5py
import torch
from torch_geometric.data import Data, Dataset

class MolecularDataset(Dataset):
    def __init__(self, root_dir):
        """
        Initialize the dataset with the root directory where H5 files are stored.
        Load all H5 files and store them in a data list.
        """
        super().__init__()
        self.root_dir = root_dir
        self.data_list = []

        # Traverse the directory structure and load each H5 file into a Data object
        for rank_dir in sorted(os.listdir(root_dir)):
            rank_path = os.path.join(root_dir, rank_dir)
            if os.path.isdir(rank_path):
                for file_name in sorted(os.listdir(rank_path)):
                    if file_name.endswith(".pt"):
                        file_path = os.path.join(rank_path, file_name)
                        data = self._load_data(file_path)
                        self.data_list.append(data)

    def _load_data(self, file_path):
        """
        Load a single H5 file and convert it to a PyTorch Geometric Data object.
        """
        with h5py.File(file_path, 'r') as h5file:
            pos = torch.tensor(h5file["pos"][:], dtype=torch.float32)
            z = torch.tensor(h5file["z"][:], dtype=torch.long)
            frame_R = torch.tensor(h5file["frame_R"][:], dtype=torch.float32)
            frame_t = torch.tensor(h5file["frame_t"][:], dtype=torch.float32)
            y = torch.tensor(h5file["y"][:], dtype=torch.float)


        return Data(pos=pos, z=z, y=y, frame_R=frame_R, frame_t=frame_t)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        """
        Get a data object from the data list.
        """
        return self.data_list[idx]

root_dir = "./data/qm9_align_h5/"
dataset = MolecularDataset(root_dir=root_dir)

# Access individual data points
print(dataset)
print(dataset[0])
