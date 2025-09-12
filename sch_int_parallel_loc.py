####################################################################
####################################################################

## Helper Classes for the Python 3 version of the Schwinger Model 
## on the Interval Hamiltonian Truncation Code

## By James Ingoldby and Rachel Houtz
## Last updated: 22nd July, 2025

###################################################################
###################################################################

#import math
import numpy as np
#import scipy as sc
import scipy.sparse as sp
#import time
#import os

from numpy import pi, sqrt
from collections import defaultdict
from collections.abc import Callable
from joblib import Parallel, delayed



"""
Include some global helper functions that will be useful for both 
the Basis and Hamiltonian classes
"""

def omega(n, m, L):
    """
    Calculate the energy of a single mode for a given 
    mode number n, mass m, and length L.

    Parameters:
    n (int): Mode number.
    m (float): Mass parameter.
    L (float): Length of the interval.

    Comment: Assumes Neveu-Schwarz boundary conditions for the fermions.

    Returns:
    float: Calculated energy.
    """
    return sqrt(((n + 1/2) * pi / L) ** 2 + m ** 2)




class Basis():
    """
    A class to build and store the H0 eigenbasis for the Schwinger model.
    """
    def __init__(self, Emax, L, M, Q=0):
        """
        Initialize the Basis class with parameters for the Schwinger model.

        Parameters:
        Emax (float): Maximum energy for the basis.
        L (float): Length of the interval.
        M (float): Mass parameter.
        Q (int): Charge parameter, default is 0.

        Outputs:
        stateList (list): List of basis states.
        size (int): Size of the basis.
        stateDict (dict): Dictionary mapping states to indices.
        """
        self.Emax = Emax
        self.L = L
        self.M = M
        self.Q = Q

        self.lmax = self.lmaxEff()
        if self.lmax < 0:
            raise ValueError("Max mode number is negative, check your parameters.")
        
        self.bitlength = self.lmax + 1
        self.shift = int(self.bitlength) 
        """ 
        Wrapping with int() above is essential to fix subtle bug. 
        States must be stored as python integers rather than numpy.int64s to prevent overflows 
        which can occur for large basis sizes.
        """


        self.fermion_energies = np.array([omega(i, self.M, self.L) for i in range(self.lmax+1)], dtype=float)

        self.stateDict = self.build_valid_states()
        self.stateList = list(self.stateDict.keys())
        self.size      = len(self.stateList)
    

    
    # Include some helper functions

    def stateQ(self, state):
        """
        Calculate the charge of a given state.

        Parameters:
        state (list): A pair of lists representing the state.

        Returns:
        int: Charge of the state.
        """
        return sum(state[0]) - sum(state[1])
    
    
    def stateEnergy(self, state):
        """
        Calculate the energy of a given state.

        Parameters:
        state (list): A pair of lists representing the state.

        Returns:
        float: Energy of the state.
        """
        f_occ = np.array(state[0], dtype=int)
        af_occ = np.array(state[1], dtype=int)
        return np.dot(f_occ, self.fermion_energies) + np.dot(af_occ, self.fermion_energies)
   
   
    def lmaxEff(self):
        """
        Calculate the effective maximum mode number based on the energy limit.

        Returns:
        int: Effective maximum mode number.
        """
        if self.Q == 0:
            remaining_energy = self.Emax - omega(0, self.M, self.L)
        else:
            remaining_energy = self.Emax - sum([omega(j, self.M, self.L) for j in range(abs(self.Q)-1)])
        return np.floor((self.L / pi) * sqrt(remaining_energy ** 2 - self.M ** 2) - 1/2).astype(int)
    
    def bits_to_int(self, bits):
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val  # Force native int, avoid NumPy propagation


    def int_to_bits(self, n, length):
        return [int(x) for x in bin(n)[2:].zfill(length)]

    def pack_state(self, state):
        f_bits, af_bits = state
        f_int = self.bits_to_int(f_bits)
        af_int = self.bits_to_int(af_bits)
        return (f_int << self.shift) | af_int

    def unpack_state(self, key):
        af_mask = (1 << self.shift) - 1
        af_int = key & af_mask
        f_int = key >> self.shift
        return [
            self.int_to_bits(f_int, self.bitlength),
            self.int_to_bits(af_int, self.bitlength)
        ]
    

    # Finally include the function to build the basis
    def genBasis(self, leff, emax, q):
        """
        Generate the basis states for a given effective maximum mode number, energy limit, and charge.
        The final answer takes the form of a list of pairs of lists, where the first list contains 
        the occupied modes and the second list contains the anti-occupied modes.

        Parameters:
        leff (int): Effective maximum mode number.
        emax (float): Maximum energy for the basis.
        q (int): Charge parameter.

        Returns:
        list: List of basis states.
        """

        result = []
        if leff == 0:
            upper_limit = int(np.floor(min(1, 1+q, (1/2) * (emax/(omega(0, self.M, self.L)) + q)))) + 1
            lower_limit = int(max(0, q))
            result = [ [[n],[n-q]] for n in range(lower_limit, upper_limit) ]
        else:
            ulimn = int(min(1, emax/omega(leff, self.M, self.L)))+1
            for n in range(ulimn):
                ulimnbar = int(min(1, emax/omega(leff, self.M, self.L) - n))+1
                for nbar in range(ulimnbar):
                    prevlist = self.genBasis(leff-1, emax-(n+nbar)*omega(leff, self.M, self.L), q - n + nbar)

                    result.extend([ [state[0] + [n], state[1] + [nbar]] for state in prevlist ])
        return result
    
    # Include function to build the valid states
    def build_valid_states(self):
        """
        Generate the valid states for the Schwinger model as pairs of lists.
        Then convert the pairs of lists into a unique integer key for each state.
        """
        list_of_states = self.genBasis(self.lmax, self.Emax, self.Q)
        # Build the dictionary
        state_to_index = {}
        for i, state in enumerate(list_of_states):
            key = self.pack_state(state)
            state_to_index[key] = i

        return state_to_index

    
    # Function to query: Given an input integer state, return the index of the state in the basis.
    def get_index(self, state):
        key = self.pack_state(state)
        return self.stateDict.get(key, -1)
    
    # Functions to convert an integer to a list of bits of a given length
    

class FermionOperators:
    """
    A class to define fermion and antifermion operators for the Schwinger model,
    as well as to compose operators for specific combinations used in the model.
    """
    def __init__(self):
        self._build_gam_terms()
    
    # Fermion creation operator a†_i
    def a_dag(self, i):
        def op(state):
            if state is None:
                return None
            (f, af), amp = state
            if f[i] == 1:
                return None  # already occupied
            sign = (-1) ** sum(f[:i])
            new_f = f.copy()
            new_f[i] = 1
            return (new_f, af), amp * sign
        return op

    # Fermion annihilation operator a_i
    def a(self, i):
        def op(state):
            if state is None:
                return None
            (f, af), amp = state
            if f[i] == 0:
                return None  # nothing to annihilate
            sign = (-1) ** sum(f[:i])
            new_f = f.copy()
            new_f[i] = 0
            return (new_f, af), amp * sign
        return op

    # Antifermion creation operator b†_i
    def b_dag(self, i):
        def op(state):
            if state is None:
                return None
            (f, af), amp = state
            if af[i] == 1:
                return None
            sign = (-1) ** (sum(f) + sum(af[:i]))
            new_af = af.copy()
            new_af[i] = 1
            return (f, new_af), amp * sign
        return op

    # Antifermion annihilation operator b_i
    def b(self, i):
        def op(state):
            if state is None:
                return None
            (f, af), amp = state
            if af[i] == 0:
                return None
            sign = (-1) ** (sum(f) + sum(af[:i]))
            new_af = af.copy()
            new_af[i] = 0
            return (f, new_af), amp * sign
        return op
    
    # Composes a list of operators and applies them in order
    def compose(self, *ops):
        def composed_op(state):
            for op in reversed(ops):  # rightmost acts first
                state = op(state)
                if state is None:
                    return None
            return state
        return composed_op
    
    # Build all γ_n operator functions dynamically
    def _build_gam_terms(self):
        """
        Dynamically builds all γ_n operator functions and stores them in self.gam_terms.
        Each function is a closure: gam_n(i, j, k, l) → operator(state) or (i, l) for 2-body terms.
        """
        def make_four_body(spec):
            def gam_fn(i, j, k, l):
                ops = []
                for (kind, dagger), idx in zip(spec, (i, j, k, l)):
                    if kind == "a":
                        ops.append(self.a_dag(idx) if dagger else self.a(idx))
                    elif kind == "b":
                        ops.append(self.b_dag(idx) if dagger else self.b(idx))
                return self.compose(*ops)
            return gam_fn

        def make_two_body(spec):
            def gam_fn(i, l):
                ops = []
                for (kind, dagger), idx in zip(spec, (i, l)):
                    if kind == "a":
                        ops.append(self.a_dag(idx) if dagger else self.a(idx))
                    elif kind == "b":
                        ops.append(self.b_dag(idx) if dagger else self.b(idx))
                return self.compose(*ops)
            return gam_fn

        self.gam_terms = {
            "g1": [make_four_body([("a", False), ("a", False), ("b", False), ("b", False)])],
            "g2": [
                make_four_body([("a", True), ("b", False), ("a", False), ("a", False)]),
                make_four_body([("b", True), ("a", False), ("b", False), ("b", False)])
            ],
            "g3": [
                make_four_body([("a", True), ("a", True), ("a", False), ("a", False)]),
                make_four_body([("b", True), ("b", True), ("b", False), ("b", False)])
            ],
            "g4": [make_four_body([("a", True), ("b", True), ("b", False), ("a", False)])],
            "g5": [make_two_body([("b", False), ("a", False)])],
            "g6": [
                make_two_body([("a", True), ("a", False)]),
                make_two_body([("b", True), ("b", False)])
            ]
        }




class Hamiltonian():
    """
    A class to build and store the Hamiltonian terms H0 and V for the Schwinger model.
    """
    def __init__(self, Emax, L, M, Q=0, n_cores=1, basis=None):
        """
        Initialize the Basis class with parameters for the Schwinger model.

        Parameters:
        Emax (float): Maximum energy for the basis.
        L (float): Length of the interval.
        M (float): Mass parameter.
        Q (int): Charge parameter, default is 0.
        n_cores (int): Number of cores available to parallelize over. Default is total minus 1.

        Outputs:
        stateList (list): List of basis states.
        size (int): Size of the basis.
        stateDict (dict): Dictionary mapping states to indices.
        """
        self.Emax = Emax
        self.L = L
        self.M = M
        self.Q = Q
        self.n_cores = n_cores

        if basis is not None:
            self.basis = basis
        else:
            self.basis = Basis(Emax, L, M, Q)
            
        self.stateList = self.basis.stateList
        self.stateDict = self.basis.stateDict
        self.dimH  = self.basis.size
        self.fermion_energies = self.basis.fermion_energies

    
    def buildH0(self):
        """
        Build the H0 Hamiltonian matrix for the Schwinger model.

        Returns:
        scipy.sparse.csr_matrix: Sparse matrix representation of H0.
        """
        occ_array = np.array([np.array(f) + np.array(af) for f, af in map(self.basis.unpack_state, self.stateList)], dtype=int)
        energy_list = occ_array @ self.fermion_energies
        self.H0 = sp.diags(energy_list, offsets=0, format='csr')

    # Include a helper function used in calculating coefficients for the V matrix
    def f(self, A: int, B: int) -> float:
        """Implements the function f[A, B] from Eq A12."""
        pi2 = pi ** 2

        if A == 0 and B == 0:
            return 1.0 / 3.0
        elif A == 0 and B != 0:
            return (-1)**(B + 1) / (pi2 * B**2)
        elif B == 0 and A != 0:
            return (-1)**(A + 1) / (pi2 * A**2)
        elif (A == B or A == -B) and A != 0:
            return 1.0 / (2 * pi2 * B**2)
        else:
            return 0.0
        
    def gencoeffs(self):
        """
        Generate the coefficients for the V Hamiltonian matrix based on the maximum mode number.

        Parameters:
        lmax (int): Maximum mode number.

        Returns:
        tuple: Lists of coefficients for different terms in the V Hamiltonian.
        """
        lmax = self.basis.lmax
        c1list, c2list, c3list, c4list, c5list, c6list = [], [], [], [], [], []

        for n in range(lmax + 1):
            for m in range(lmax + 1):
                for k in range(lmax + 1):
                    for l in range(lmax + 1):
                        val1 = self.f(n + k + 1, m + l + 1)
                        if val1 != 0:
                            c1list.append([l, k, m, n, -val1])
                        val2 = self.f(n - l, m + k + 1)
                        if val2 != 0:
                            c2list.append([l, k, m ,n, 2 * val2])

        for m in range(lmax + 1):
            for n in range(m + 1):
                for l in range(lmax + 1):
                    for k in range(l + 1):
                        if n == m == l == k:
                            continue
                        elif n == k and m == l:
                            val = 1 / (2 * pi ** 2 * (n - m) ** 2) - 1.0 / 3.0
                        elif m == l and n < k:
                            val = 2 * self.f(m - k, n - l) - 2 * self.f(n - k, m - l)
                        elif m < l:
                            val = 2 * self.f(m - k, n - l) - 2 * self.f(n - k, m - l)
                        else:
                            val = 0
                        if val != 0:
                            c3list.append([n, m, k, l, val])

        for m in range(lmax + 1):
            for k in range(lmax + 1):
                for l in range(lmax + 1):
                    for n in range(l + 1):
                        if n == l and m == k:
                            val = 1 / (2 * pi ** 2 * (n + m + 1) ** 2) - 1.0 / 3.0
                        elif n == l and m < k:
                            val = 2 * self.f(n + m + 1, k + l + 1) - 2 * self.f(n - l, m - k)
                        elif n < l:
                            val = 2 * self.f(n + m + 1, k + l + 1) - 2 * self.f(n - l, m - k)
                        else:
                            val = 0
                        if val != 0:
                            c4list.append([n, m, k, l, val])

        for n in range(lmax + 1):
            l0 = 1 if n % 2 == 0 else 0
            for l in range(l0, lmax + 1, 2):
                if n == l:
                    continue
                sum_val = 0.0
                m1 = (n - l - 1) / 2
                if m1 >= 0 and m1.is_integer():
                    sum_val += self.f(n - int(m1), int(m1) + l + 1)
                m2 = (l - n - 1) / 2
                if m2 >= 0 and m2.is_integer():
                    sum_val -= self.f(int(m2) - l, int(m2) + n + 1)
                if sum_val != 0:
                    c5list.append([l, n, sum_val])

        for l in range(lmax + 1):
            sum_diag = 1 / 6 + sum(1 / k ** 2 for k in range(1, l + 1)) / (2 * pi ** 2)
            c6list.append([l, l, sum_diag])
            for n in range(l):
                val = 2 / (pi ** 2 * (l - n) ** 2) if (l - n) % 2 == 1 else 0.0
                if val != 0:
                    c6list.append([n, l, val])

        return c1list, c2list, c3list, c4list, c5list, c6list

    def apply_tensor_operator(self, state, cxlist, operator_fns, energy_check=False, tol=1e-12):
        """
        Generic routine to apply one or more tensor-weighted operators to a state.

        Parameters:
            state: ((f_list, af_list), amplitude)
            cxlist: list of [i, j, k, l, coeff] or [i, l, coeff]
            operator_fns: a single function or list of functions
            energy_check: bool — whether to truncate based on energy
            tol: numerical tolerance for filtering

        Returns:
            List of ((f, af), amplitude) states after action
        """
        result_dict = defaultdict(float)

        if isinstance(operator_fns, Callable):
            operator_fns = [operator_fns]

        for coeff_entry in cxlist:
            *indices, coeff = coeff_entry

            for op_fn in operator_fns:
                op = op_fn(*indices)
                new_state = op(state)
                if new_state is None:
                    continue

                (f_new, af_new), amp = new_state
                amp *= coeff

                if abs(amp) < tol:
                    continue

                if energy_check:
                    energy = self.basis.stateEnergy([f_new, af_new])
                    if energy > self.Emax:
                        continue

                key = (tuple(f_new), tuple(af_new))
                result_dict[key] += amp

        return [(((list(f), list(af)), amp)) for (f, af), amp in result_dict.items() if abs(amp) > tol]

    

    def apply_all_terms(
        self,
        state,
        coeff_lists,
        operator_sets,
        energy_flags,
        tol=1e-12
    ):
        """
        Apply all interaction terms (g_n) to a single state and return the result.

        Parameters:
            state: ((f_list, af_list), amplitude)
            coeff_lists: dict of {gn_label: coefficient_list}
            operator_sets: dict of {gn_label: operator_generator or list thereof}
            energy_flags: dict of {gn_label: bool}
            tol: numerical tolerance

        Returns:
            List of ((f, af), amplitude) states
        """
        result_dict = defaultdict(float)

        for label, cxlist in coeff_lists.items():
            gam_fns = operator_sets[label]
            energy_check = energy_flags.get(label, False)

            term_results = self.apply_tensor_operator(
                state,
                cxlist,
                gam_fns,
                energy_check=energy_check,
                tol=tol
            )

            for ((f, af), amp) in term_results:
                key = (tuple(f), tuple(af))
                result_dict[key] += amp

        return [(((list(f), list(af)), amp)) for (f, af), amp in result_dict.items() if abs(amp) > tol]
    
    def _process_state_column_static(self, col_index, packed, basis, coeff_lists, operator_sets, energy_check_flags, Emax, tol):
        unpacked_state = basis.unpack_state(packed)
        state = (unpacked_state, 1.0)

        # Use a local apply_all_terms helper
        results = []

        for label, cxlist in coeff_lists.items():
            gam_fns = operator_sets[label]
            energy_check = energy_check_flags.get(label, False)

            if isinstance(gam_fns, Callable):
                gam_fns = [gam_fns]

            for entry in cxlist:
                op = None
                if len(entry) == 5:
                    i, j, k, l, coeff = entry
                    for gam in gam_fns:
                        op = gam(i, j, k, l)
                        new_state = op(state)
                        if new_state is None:
                            continue
                        (f, af), amp = new_state
                        amp *= coeff
                        if abs(amp) < tol:
                            continue
                        if energy_check and basis.stateEnergy([f, af]) > Emax:
                            continue
                        results.append((tuple(f), tuple(af), col_index, amp))
                elif len(entry) == 3:
                    i, l, coeff = entry
                    for gam in gam_fns:
                        op = gam(i, l)
                        new_state = op(state)
                        if new_state is None:
                            continue
                        (f, af), amp = new_state
                        amp *= coeff
                        if abs(amp) < tol:
                            continue
                        if energy_check and basis.stateEnergy([f, af]) > Emax:
                            continue
                        results.append((tuple(f), tuple(af), col_index, amp))

        return results

    def buildV(self, start=0, end=None):
        """
        Build the V Hamiltonian matrix for the Schwinger model over a slice of basis states.

        Parameters:
            start (int): Starting index in basis.stateList (inclusive).
            end (int): Ending index in basis.stateList (exclusive). If None, goes to end.

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix representation of V slice.
        """
        # Instantiate operator sets
        ops = FermionOperators()
        c1, c2, c3, c4, c5, c6 = self.gencoeffs()

        coeff_lists = {
            "g1": c1,
            "g2": c2,
            "g3": c3,
            "g4": c4,
            "g5": c5,
            "g6": c6,
        }

        operator_sets = {
            "g1": ops.gam_terms["g1"],
            "g2": ops.gam_terms["g2"],
            "g3": ops.gam_terms["g3"],
            "g4": ops.gam_terms["g4"],
            "g5": ops.gam_terms["g5"],
            "g6": ops.gam_terms["g6"],
        }

        energy_check_flags = {
            "g1": False,
            "g2": True,
            "g3": True,
            "g4": True,
            "g5": False,
            "g6": True,
        }

        # Set end to full basis size if not provided
        if end is None:
            end = len(self.basis.stateList)

        # Prepare args for parallel map over the slice
        args = [
            (i, packed, self.basis, coeff_lists, operator_sets, energy_check_flags, self.Emax, 1e-12)
            for i, packed in enumerate(self.basis.stateList[start:end], start)
        ]

        # Use joblib to parallelize over this slice
        results = Parallel(n_jobs=self.n_cores)(
            delayed(self._process_state_column_static)(*arg)
            for arg in args
        )

        # Flatten results and convert to rows, cols, data
        rows, cols, data = [], [], []
        for termlist in results:
            for f, af, col, amp in termlist:  # type: ignore
                packed_row = self.basis.pack_state([list(f), list(af)])
                row_index = self.basis.stateDict.get(packed_row, -1)
                if row_index != -1:
                    rows.append(row_index)
                    cols.append(col)
                    data.append(amp)

        dim = self.dimH
        self.V = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
        