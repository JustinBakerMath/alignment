from torch_canon.pointcloud import CanonEn as Canon

import argparse
import sys
import logging

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import wasserstein_distance_nd

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from ase.calculators.lj import LennardJones
from ase.spacegroup import crystal
import numpy as np

import spglib


calc = LennardJones()


a = 3.57
atoms = crystal('C', [(0, 0, 0)], spacegroup=227,
                  cellpar=[a, a, a, 90, 90, 90])


lattice = atoms.cell.array  # Get the lattice vectors
positions = atoms.positions   # Get atomic positions
print(positions)
numbers = atoms.numbers       # Get atomic numbers

# Use spglib to find the space group
try:
    # Create a structure for spglib
    structure = (lattice, positions, numbers)
    space_group_info = spglib.get_spacegroup(structure)
    space_group = space_group_info[0]  # Get the space group symbol
    #print(f"Space Group: {space_group_info}")
    if space_group_info != 'P1 (1)':
        print(space_group_info, type(space_group_info))
        print(f"Space Group: {space_group_info}")
except Exception as e:
    print(f"Error determining space group: {e}")

frame = Canon(tol=1e-6)

normalized_positions, frame_R, frame_t = frame.get_frame(atoms.positions, atoms.numbers)

atoms_aligned = atoms.copy()
atoms_aligned.positions = normalized_positions

from ase.visualize import view

atoms_asu = atoms.copy()

list_of_asu_atoms = [5, 1, 0, 6]
# Assign tags to differentiate the atoms
# Let's assign tag=1 to four atoms and tag=2 to the other four
for i in range(len(atoms_asu)):
    if i in list_of_asu_atoms:
        atoms_asu.numbers[i] = 8
    else:
        atoms_asu.numbers[i] = 10

# _ = atoms_to_html(atoms_asu, "diamond_asu.html")
# Visualize with color by tag
view(atoms_asu, viewer="x3d")


ASUN = Canon(tol=1e-8, save = True)
_ = ASUN.get_frame(atoms.positions, atoms.numbers)

print(ASUN.symmetric_elements)
print(ASUN.simple_asu)

for i in range(len(atoms_asu)):
    if i in ASUN.simple_asu:
        atoms_asu.numbers[i] = 8
    else:
        atoms_asu.numbers[i] = 10

# _ = atoms_to_html(atoms_asu, "diamond_asu_from_our_code.html")

view(atoms_asu, viewer="x3d")
