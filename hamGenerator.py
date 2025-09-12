import argparse
import time
import os
import numpy as np
import scipy.sparse as sp
from joblib import cpu_count
from sch_int_parallel_loc import Basis, Hamiltonian

def main():
    parser = argparse.ArgumentParser(description="Build basis, free and interaction Hamiltonian of interval Schwinger model.")
    parser.add_argument("--Emax", type=float, required=True, help="Energy cutoff")
    parser.add_argument("--L", type=float, required=True, help="System size L")
    parser.add_argument("--Q", type=int, default=0, help="Charge sector")
    parser.add_argument("--n_cores", type=int, default=-1, help="Number of CPU cores for parallelisation")
    parser.add_argument("--out", type=str, required=True, help="Output filename for sparse matrix (.npz)")
    args = parser.parse_args()

    # Repeat parameters back to user for comprehension
    print("Building matrix representation of interaction for parameters:\n")
    print(f"Cutoff: {args.Emax}")
    print(f"Length: {args.L}")
    print(f"Charge: {args.Q}")
    print(f"Output file name: {args.out}")
    actual_cores = cpu_count() if args.n_cores == -1 else args.n_cores

    print(f"Number of Cores: {actual_cores}")

    # Generate basis states
    print("Generating basis of states...")
    basis = Basis(Emax=args.Emax, L=args.L, M=0, Q=args.Q)
    print("Total number of states is: " + str(int(basis.size)))

    # Initialize Hamiltonian
    H = Hamiltonian(Emax=args.Emax, L=args.L, M=0, Q=args.Q, n_cores=args.n_cores, basis=basis)

    # Build H0
    H.buildH0()

    # Build interaction matrix
    print("Building the Interaction...")
    start_time = time.time()
    H.buildV()
    time_diff = time.time() - start_time
    print(f"Execution time: {time_diff:.6f} seconds")

    # Save the data to .npz file
    folder = "Hams"
    os.makedirs(folder, exist_ok=True)

    # Create filename
    filename = os.path.join(folder, args.out)

    np.savez_compressed(
    filename,
    basis=np.array(basis, dtype=object),  # cast to array for npz format
    H0_data=H.H0.data,
    H0_indices=H.H0.indices, # type: ignore
    H0_indptr=H.H0.indptr,   # type: ignore
    H0_shape=H.H0.shape,
    V_data=H.V.data,
    V_indices=H.V.indices,
    V_indptr=H.V.indptr,
    V_shape=H.V.shape
    )


if __name__ == "__main__":
    main()
