python ./examples/expl_frame/00_qm9.py --n_data 10
python ./examples/expl_frame/01_canon_qm9.py --n_data 10


mpiexec -n 4 python ./examples/spmd/00_qm9.py --n_data 10
mpiexec -n 4 python ./examples/spmd/01_canon_qm9.py --n_data 10
