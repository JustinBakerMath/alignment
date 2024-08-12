mkdir -p ./out/
rm -rf ./out/qm9canon_measurements ./out/qm9canon_database
hpcrun -t -o ./out/qm9canon_measurements mpiexec -n 10 python3 ./examples/spmd/01_canon_qm9.py --n_data 2000 --n_batch 16
hpcstruct --gpucfg yes ./out/qm9canon_measurements
hpcprof ./out/qm9canon_measurements -o ./out/qm9canon_database
hpcviewer ./out/qm9canon_database
