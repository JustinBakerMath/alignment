"""
QM7 (Quantum Mechanics 7)

A collection of molecules with up to nine heavy atoms (C, O, N, S)
used as a benchmark dataset for molecular property prediction and
graph-classification tasks.

This file is a loader for variations of the dataset.

"""
import os
from typing import Optional

import h5py
from rdkit import Chem
from tqdm import tqdm

import numpy as np
import torch

from e3nn.o3 import Irreps, spherical_harmonics
from pointgroup import PointGroup

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree

from torch_scatter import scatter

from torch_canon.E3Global.CategoricalPointCloud import CatFrame as Frame

from pointgroup import PointGroup

point_groups = {'C1':0, 'C1h':1, 'C1v':2, 'C2':3, 'C2d':4, 'C2h':5, 'C2v':6, 'C3':7, 'C3h':8, 'C3v':9, 'C4':10, 'Cs':11, 'Cinfv':12, 'Ci':13, 'Dinfh':14, 'D2':15, 'D2d':16, 'D2h':17, 'D3':18, 'D3d':19, 'D3h':20, 'D6h':21, 'Oh':22, 'Td':23, 'S2':24, 'S4':25}

targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0',
                      'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']


pg_order = {
	"C1": 1,   # Only the identity element
    "C1v": 2,
    "C1h": 2,
	"Cs": 2,   # Identity + one mirror plane
	"Ci": 2,   # Identity + inversion center
    "Cinfv": 2,
	"C2": 2,   # One 2-fold rotation axis
    "C2h": 2*2,
    "C2d": 2*2,
	"C3": 3,   # One 3-fold rotation axis
	"C4": 4,   # One 4-fold rotation axis
	"C5": 5,   # One 4-fold rotation axis
	"C6": 6,   # One 6-fold rotation axis
    "C7": 7,
    "C8": 8,
    "C9": 9,
    "Dinfh": 24,
	"D2": 4,   # Three 2-fold axes
    "D2d": 8,
    "D2h": 8,
	"D3": 6,   # One 3-fold axis + two 2-fold axes
    "D3d": 12,
    "D3h": 12,
	"D4": 8,   # One 4-fold axis + four 2-fold axes
    "D4d": 16,
    "D4h": 16,
    "D5": 10,
    "D5d": 20,
    "D5h": 20,
	"D6": 12,  # One 6-fold axis + six 2-fold axes
	"D6d": 4*6,
    "D6v": 4*6,
    "D9": 2*9,
    "D9h": 4*9,
    "D9d": 4*9,
	"C2v": 4,  # C2 axis + two vertical mirror planes
	"C3v": 6,  # C3 axis + three vertical mirror planes
	"C4v": 8,  # C4 axis + four vertical mirror planes
    "C5v": 10, # C5 axis + five vertical mirror planes
	"C6v": 12, # C6 axis + six vertical mirror planes
    "C7v": 2*7,
    "C9v": 2*9,
    "C9h": 2*9,
	"S4": 4,   # Four-fold improper rotation axis
	"S6": 6,   # Six-fold improper rotation axis
	"D2h": 8,  # D2 group + a horizontal mirror plane
	"D3h": 12, # D3 group + a horizontal mirror plane
	"D4h": 16, # D4 group + a horizontal mirror plane
	"D6h": 24, # D6 group + a horizontal mirror plane
	"Td": 24,  # Tetrahedral symmetry
	"Oh": 48,  # Octahedral symmetry
	"Ih": 120,  # Icosahedral symmetry
    "S2": 2,
    "S12": 12,
}



atomic_number_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16:'S', 17:'Cl'}
atomic_symbol_to_number = {val:key for key,val in atomic_number_to_symbol.items()}

portions = {
    'fixed':(50,100),
    }


broken_data_idx = [2611, 2610, 2610, 14692, 15475, 15480, 15484, 15486, 15497, 4411, 15542, 17032, 20101, 740, 28389, 28398, 28399, 28400, 28420, 28481, 28622, 28626, 33533, 33560, 33566, 33567, 33571, 33572, 33579, 33582, 33589, 33600, 33602, 33604, 33606, 33618, 33629, 40551, 40601, 45209, 45214, 45223, 45259, 45264, 45276, 45280, 45306, 45351, 45354, 45356, 45360, 45363, 45365, 45366, 45367, 45369, 45371, 45373, 45374, 45375, 45378, 45379, 45385, 45386, 45388, 45390, 45396, 45400, 45402, 45407, 45408, 45409, 45411, 45416, 45418, 45419, 45421, 45423, 45425, 45427, 45429, 45431, 45433, 45442, 45446, 51940, 51953, 51990, 52014, 65965, 65969, 65985, 65990, 65991, 65994, 65997, 65999, 66000, 66004, 66010, 66012, 66017, 66018, 66019, 66022, 66030, 66031, 66038, 66481, 67617, 67743, 71537, 71590, 71608, 71827, 71911, 76980, 76997, 77033, 77385, 77520, 81758,
82843, 82866, 85680, 85731, 85748, 94167, 95143, 95147, 95148, 95150, 95152, 95154, 95156, 95160, 95166, 95167, 95168, 95172, 95177, 95182, 95187, 95189, 95191, 95193, 95195, 95196, 95197, 95198, 95204, 95207, 95215, 95217, 95218, 95219, 95220, 95223, 95224, 95227, 95231, 95232, 95234, 95235,
205433, 206242, 206481, 211261, 212334, 212402, 212483, 219372, 219401, 219439, 223537, 223554, 223559, 235137, 235139, 235144, 235160, 235174, 235214, 254935, 254942, 255148, 261666, 262195, 264761, 278760, 278793, 278799, 278803, 278810, 278816, 278819, 278825, 304598, 304615, 305737, 305765, 305783, 305803,
 307976, 307982, 308068, 314847, 314875, 314876, 314905, 315553, 315560, 315624, 321286, 329562, 330289, 330398, 330420, 330451, 330453, 330821, 331086, 331096, 331167, 331170, 331178, 340380, 340415, 342507, 354458, 354678, 354707, 358954, 358966, 367847, 372835, 373426, 375840, 378668, 378700,
 378727, 380288, 380309, 380317, 380403, 381604, 381660, 383105, 383143, 383201, 383213, 383214, 383219, 383247, 383263
 ]

broken_data = [
['C', 'C', 'C', 'C', 'N', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'N', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['O', 'C', 'C', 'N', 'C', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'H', 'H'],
['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H'],
['C', 'C', 'C', 'N', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'N', 'H'],
['N', 'C', 'C', 'N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['N', 'C', 'C', 'C', 'C', 'N', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'N', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'N', 'C', 'N', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'N', 'N', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'N', 'N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'H', 'H'],
['C', 'C', 'C', 'C', 'N', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'N', 'C', 'C', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'C', 'N', 'H'],
['C', 'C', 'C', 'O', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['O', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['O', 'C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'H', 'H', 'H', 'H'],
['C', 'O', 'C', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'O', 'C', 'C', 'O', 'H', 'H', 'H', 'H'],
['O', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
['O', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'O', 'C', 'C', 'H', 'H'],
['C', 'C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['O', 'C', 'C', 'C', 'C', 'O', 'H', 'H'],
['C', 'C', 'C', 'N', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['O', 'C', 'C', 'C', 'C', 'N', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'C', 'C', 'N', 'N', 'S', 'H', 'H'],
['C', 'C', 'C', 'N', 'S', 'C', 'N', 'H', 'H'],
['C', 'C', 'S', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['Cl', 'C', 'C', 'N', 'C', 'C', 'Cl', 'H', 'H', 'H'],
['C', 'C', 'C', 'S', 'O', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['N', 'C', 'C', 'C', 'S', 'C', 'C', 'H', 'H', 'H'],
['N', 'C', 'C', 'C', 'S', 'N', 'C', 'H', 'H'],
['N', 'S', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
['O', 'C', 'C', 'C', 'S', 'N', 'C', 'H', 'H', 'H', 'H', 'H'],
['O', 'S', 'O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
['O', 'S', 'O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
['C', 'C', 'S', 'O', 'O', 'C', 'C', 'H', 'H'],
['S', 'C', 'N', 'C', 'C', 'N', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
]


"""
================
Generate Loaders
================
"""




def qm7_loaders(loader_cfg: dict) -> dict:

    featurization = 'atomic-nubmer'
    adjacency = 'bond'
    split = 'fixed'
    batch_size = loader_cfg['batch_size']
    pre_transform = []
    transform = []

    pre_transform.append(T.RadiusGraph(8.0))
    pre_transform.append(Degree())

    if 'align' in loader_cfg:
        align_cfg = loader_cfg['align']
        pre_transform.append(Align(tol=align_cfg['tol']))
        pth = './data/qm7x_align'
    else:
        pth = './data/qm7x'

    transform.append(GetTarget(loader_cfg['target']))
    
    dataset = QM7X(root=pth, pre_transform=T.Compose(pre_transform), transform=T.Compose(transform))


    degrees = []
    lst_dataset = []
    lst_dataset_idx = []
    restart = 0
    k = 5

    for i,data in tqdm(enumerate(dataset), total=len(dataset)):

        symbols = [atomic_number_to_symbol[atomic_num.item()] for atomic_num in data.z]
        if symbols in broken_data[k-3:k-2]:
            print('*'*50)
            symbols = [atomic_number_to_symbol[atomic_num.item()] for atomic_num in data.z]
            print(symbols)
            print(data.hVDIP)
            print(data.pos)
            pg = PointGroup(data.pos, [atomic_number_to_symbol[atomic_num.item()] for atomic_num in data.z],0.03,8).get_point_group()
            print(pg)
            pg_hVDIP = PointGroup(data.hVDIP, [atomic_number_to_symbol[atomic_num.item()] for atomic_num in data.z],0.01,4).get_point_group()
            print(pg_hVDIP)
            degrees.append(data.degrees.tolist()[0])
        if i>broken_data_idx[k]:
            break

    degrees_hist = torch.from_numpy(np.histogram(degrees, bins=range(10))[0])
    #dataset = FilteredQM9('',dataset[lst_dataset], transform=T.Compose(transform))
    dataset = dataset[broken_data_idx]
    exit()
    mean = 0
    std = 1

    train_dataset, val_dataset, test_dataset = fixed_splits(dataset,[])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader, 'mean': mean, 'std': std, 'degrees_hist': degrees_hist}

"""
===========
Data Splits
===========

    Split the QM9 dataset containing 130,831 molecules into
    training, validation and test sets:

    `fixed_splits`: This follows the splits of EGNN and Cormorant
     - trainset <- portions either 110_000, 100_000 or 50_000 molecules
     - testset <- 0.1 * (QM9 total number of molecules)
     - valset <- remaining molecules
    
"""
def fixed_splits(dataset, broken_data_idx):
    if len(broken_data_idx) == 0:
        train_split = int(0.6*len(dataset))
        val_split = int(0.2*len(dataset))
        return dataset[:train_split], dataset[train_split:train_split+val_split], dataset[train_split+val_split:]
    else:
        broken_data = dataset[broken_data_idx]
        not_broken_data_idx = [i for i in range(len(dataset)) if i not in broken_data_idx]
        symm_data = dataset[not_broken_data_idx]

        shuffle_broken = np.random.permutation(len(broken_data_idx))
        shuffle_symm = np.random.permutation(len(dataset)-len(broken_data_idx))

        broken_data = broken_data[shuffle_broken]
        symm_data = symm_data[shuffle_symm]

        dataset = [data for data in broken_data] + [data for data in symm_data]

        train_split = int(0.6*len(dataset))
        val_split = int(0.2*len(dataset))
        test_split = len(dataset) - train_split - val_split
        return dataset[:train_split], dataset[train_split:train_split+val_split], dataset[train_split+val_split:]

"""
Transformations 
~~~~~~~~~~~~~~~

"""

class Degree(T.BaseTransform):
    def __call__(self, data):
        data.degrees = degree(data.edge_index[0], num_nodes=data.z.shape[0])
        return data

class Align(T.BaseTransform):
    def __init__(self, tol: Optional[float] = 1e-3):
        self.tol = tol

    def __call__(self, data):
        frame = Frame(tol=self.tol, save=True)
        align_pos, frame_R, frame_t = frame.get_frame(data.pos, data.z)
        data.align_pos = torch.from_numpy(align_pos)
        data.frame_R = frame_R
        data.frame_t = frame_t
        symmetric_elements = frame.symmetric_elements
        symmetric_edge_index = [[i,j] for symmetry_element in symmetric_elements for i in symmetry_element for j in symmetry_element]
        projection_edge_index = [[symmetry_element[0],symmetry_element[j]] for symmetry_element in symmetric_elements for j in range(1,len(symmetry_element))]
        data.project_edge_index = torch.tensor(projection_edge_index, dtype=torch.long).T
        data.symmetric_edge_index = torch.tensor(symmetric_edge_index, dtype=torch.long).T
        asu = frame.simple_asu
        asu_edge_index = [[i,j] for i in asu for j in asu]
        data.asu_edge_index = torch.tensor(asu_edge_index, dtype=torch.long).T
        return data

class GetTarget(T.BaseTransform):
    def __init__(self, target: str):
        self.target = target

    def __call__(self, data):
        if self.target == 'hVDIP':
            data.y = data.hVDIP
        elif self.target == 'HLgap':
            data.y = data.HLgap
        else:
            raise ValueError(f"Invalid target: {self.target}")
        return data

class IntegerEdgeFeatures(T.BaseTransform):
    def __call__(self, data):
        data.edge_attr = data.edge_attr[:, 0].to(torch.long)
        return data

class GraphToMol(T.BaseTransform):
    def __call__(self, data):
        mol = Chem.RWMol()

        # Add atoms to the molecule using data.z (atomic numbers)
        for atomic_num in data.z:
          atom = Chem.Atom(int(atomic_num.item()))  # Convert to RDKit atom
          mol.AddAtom(atom)

        # Add bond information based on distance thresholds or predefined bond data
        # Example: adding bonds based on distance threshold (simple nearest neighbor)
        threshold = 1.6  # Threshold distance for bond formation

        for i in range(len(data.pos)):
          for j in range(i + 1, len(data.pos)):
              dist = torch.norm(data.pos[i] - data.pos[j]).item()
              if dist < threshold:
                  mol.AddBond(i, j, Chem.BondType.SINGLE)  # Add single bond for simplicity

        # Convert to a Mol object
        data.mol = mol.GetMol()
        return data

class AddPG(T.BaseTransform):
    def __call__(self, data):
        try:
            symbols = [atomic_number_to_symbol[atomic_num.item()] for atomic_num in data.z]
            pg = PointGroup(data.pos, symbols).get_point_group()
            data.pg = pg
        except:
            data.pg = 'C1'
        return data

class GetPG(T.BaseTransform):
    def __call__(self, data):
        data.y = torch.tensor([point_groups[data.pg]],dtype=torch.long)
        return data
    

class FilteredQM9(InMemoryDataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        self.data_list = data_list
        super(FilteredQM9, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(self.data_list)  # Collate the list into a usable format

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]  # This allows you to access the data points as you would in the original QM9 dataset

class QM7X(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.hdf5')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []

        for hdf5_file in tqdm(self.raw_paths, desc="Processing HDF5 Files", unit="file"):
            with h5py.File(hdf5_file, 'r') as f:
                # Collect all relevant datasets for tqdm progress monitoring
                dataset_paths = []
                
                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Group):
                        if all(key in obj for key in ['atNUM', 'atXYZ', 'hVDIP']):
                            dataset_paths.append(obj.name)

                # Visit all groups and collect paths containing required datasets
                f.visititems(collect_datasets)
                
                # Use tqdm to process each valid group path
                for path in tqdm(dataset_paths, desc="Processing HDF5 Data", unit="group"):
                    group = f[path]

                    # Extract data from HDF5 group
                    atomnum = group['atNUM'][:]
                    pos = group['atXYZ'][:]
                    dipole = group['hVDIP'][:]
                    gap = group['HLgap'][:]
                    
                    
                    # Convert data to torch tensors
                    atomnum = torch.tensor(atomnum, dtype=torch.long)
                    unique_atomnum = torch.unique(atomnum)
                    assert unique_atomnum.size(0) <= 5, "All atoms in a molecule must be of the same type"
                    pos = torch.tensor(pos, dtype=torch.float)
                    dipole = torch.tensor(dipole, dtype=torch.float)
                    gap = torch.tensor(gap, dtype=torch.float)
                    
                    # Create PyTorch Geometric Data object
                    data = Data(z=atomnum, pos=pos, hVDIP=dipole, HLgap=gap)

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
        

        # Save data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Instantiate and process the dataset
#dataset = QM7X(root='./data/qm7x/', force_reload=True)
#print(len(dataset))
#print(dataset[0])
