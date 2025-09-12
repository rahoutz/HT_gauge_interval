####################################################################
####################################################################

## Helper Classes for the Python 3 version of the Nonabelian Gauge Theory 
## on the Interval Hamiltonian Truncation Code

## By James Ingoldby and Rachel Houtz
## Last updated: 31st July, 2025

###################################################################
###################################################################

#import math
import numpy as np
#import scipy as sc
import scipy.sparse as sp
#import time
#import os

from itertools import product
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
    A class to build and store the H0 eigenbasis for interval nonabelian gauge theory.
    """
    def __init__(self, Emax, L, M, Nc, B=0):
        """
        Initialize the Basis class with parameters for the gauge theory.

        Parameters:
        Emax (float): Maximum energy for the basis.
        L (float): Length of the interval.
        M (float): Mass parameter.
        Nc (int): Number of colours.
        B (int): Baryon number, default is 0.

        Outputs:
        stateList (list): List of basis states.
        size (int): Size of the basis.
        stateDict (dict): Dictionary mapping states to indices.
        """
        self.Emax = Emax
        self.L = L
        self.M = M
        self.Nc = Nc
        self.B = B

        self.lmax = self.lmaxEff()
        if self.lmax < 0:
            raise ValueError("Max mode number is negative, check your parameters.")
        
        self.bitlength = self.lmax + 1 # Number of bits needed to represent each mode occupation. Used when packing states

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
        Calculate the total energy of a nonabelian basis state.

        Parameters:
        state (tuple): A tuple of Nc elements, each being a pair of lists
                       representing [fermion_occupations, antifermion_occupations].

        Returns:
        float: Total energy of the Nc-color state.
        """
        total_energy = 0.
        for color_state in state:
            f_occ = np.array(color_state[0], dtype=int)
            af_occ = np.array(color_state[1], dtype=int)
            total_energy += np.dot(f_occ, self.fermion_energies)
            total_energy += np.dot(af_occ, self.fermion_energies)
        return total_energy
   
   
    def lmaxEff(self):
        """
        Calculate the effective maximum mode number based on the energy limit.

        Returns:
        int: Effective maximum mode number.
        """
        if self.B == 0:
            remaining_energy = self.Emax - omega(0, self.M, self.L)
        else:
            energy_from_antifermions = sum([omega(j, self.M, self.L) for j in range(abs(self.B)-1)])
            energy_from_other_colors = (self.Nc - 1) * sum([omega(j, self.M, self.L) for j in range(abs(self.B))])
            remaining_energy = self.Emax - energy_from_antifermions - energy_from_other_colors
        return np.floor((self.L / pi) * sqrt(remaining_energy ** 2 - self.M ** 2) - 1/2).astype(int)
    

    def pack_state(self, state):
        """
        Pack a tuple of Nc pairs of bitlists into a bytearray.

        Parameters:
        state (tuple): Each element is [f_bits, af_bits] for a color.

        Returns:
        bytes: Packed byte representation of the state.
        """
        bitstream = []
        for f_bits, af_bits in state:
            bitstream.extend(f_bits)
            bitstream.extend(af_bits)

        # Pad to nearest byte
        pad_len = (8 - len(bitstream) % 8) % 8
        bitstream.extend([0] * pad_len)

        # Group bits into bytes
        byte_vals = []
        for i in range(0, len(bitstream), 8):
            byte = 0
            for b in bitstream[i:i+8]:
                byte = (byte << 1) | b
            byte_vals.append(byte)

        return bytes(byte_vals)
    
    def unpack_state(self, byte_data):
        """
        Unpack a byte representation into a tuple of Nc [f_bits, af_bits].

        Parameters:
        byte_data (bytes): Packed state.

        Returns:
        tuple: Tuple of Nc elements, each a [f_bits, af_bits] list.
        """
        total_bits = 2 * self.bitlength * self.Nc
        bitstream = []

        for byte in byte_data:
            bits = [(byte >> i) & 1 for i in reversed(range(8))]
            bitstream.extend(bits)

        # Trim padding
        bitstream = bitstream[:total_bits]

        # Slice into Nc pairs
        state = []
        for i in range(self.Nc):
            start = i * 2 * self.bitlength
            f_bits = bitstream[start:start + self.bitlength]
            af_bits = bitstream[start + self.bitlength:start + 2 * self.bitlength]
            state.append([f_bits, af_bits])

        return tuple(state)


    # Finally include the function to build the basis
    def genBasis(self, leff, emax, q):
        """
        Generate the basis states of the Schwinger Model for a given effective maximum mode number, energy limit, and charge.
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
        Begin by generating the valid states for the Schwinger model as pairs of lists.
        Then take tensor products of Schwinger states to build the basis of the nonabelian gauge theory.
        Then filter out all of the states with energy greater than Emax.
        Finally, build a dictionary mapping the packed state to its index in the basis.
        """
        # Get abelian Schwinger basis
        sch_basis = self.genBasis(self.lmax, self.Emax, self.B)

        # Build tensor product basis for SU(Nc) theory (uncut)
        tensor_product_basis = list(product(sch_basis, repeat=self.Nc))

        # Filter out states with energy greater than Emax
        filtered_basis = [state for state in tensor_product_basis if self.stateEnergy(state) <= self.Emax]
        
        # Build the dictionary
        state_to_index = {}
        for i, state in enumerate(filtered_basis):
            key = self.pack_state(state)
            state_to_index[key] = i
        
        return state_to_index

    # Function to query: Given an input integer state, return the index of the state in the basis.
    def get_index(self, state):
        key = self.pack_state(state)
        return self.stateDict.get(key, -1)
    

    
class NonAbelianFermionOperators:
    def __init__(self, Nc):
        self.Nc = Nc
        self.build_gam_tau_terms()

    def count_fermions_before(self, state, op_type, color, site):
        """
        Count the number of occupied modes that come before the given
        (op_type, color, site) in your canonical ordering:
        (color0 f), (color0 af), (color1 f), (color1 af), ...

        Parameters:
        - state: tuple of Nc elements, each a pair (f_bits, af_bits)
        - op_type: "f" or "af"
        - color: int, color index
        - site: int, site index

        Returns:
        - int: number of occupied modes before this one
        """
        count = 0

        for c in range(self.Nc):
            f_bits, af_bits = state[c]

            # Modes before current color
            if c < color:
                count += sum(f_bits)
                count += sum(af_bits)

            # Modes at the same color
            elif c == color:
                if op_type == "af":
                    count += sum(f_bits)  # all fermions at this color are before all af
                    count += sum(af_bits[:site])
                else:  # op_type == "f"
                    count += sum(f_bits[:site])

        return count


    def a_dag(self, i, c):
        def op(state_amp):
            if state_amp is None:
                return None
            state, amp = state_amp
            f, af = state[c]
            if f[i] == 1:
                return None
            sign = (-1) ** self.count_fermions_before(state, "f", c, i)
            new_f = f.copy()
            new_f[i] = 1
            new_state = list(state)
            new_state[c] = (new_f, af)
            return tuple(new_state), amp * sign
        return op
    
    def a(self, i, c):
        def op(state_amp):
            if state_amp is None:
                return None
            state, amp = state_amp
            f, af = state[c]
            if f[i] == 0:
                return None  # nothing to annihilate
            sign = (-1) ** self.count_fermions_before(state, "f", c, i)
            new_f = f.copy()
            new_f[i] = 0
            new_state = list(state)
            new_state[c] = (new_f, af)
            return tuple(new_state), amp * sign
        return op
    
    def b_dag(self, i, c):
        def op(state_amp):
            if state_amp is None:
                return None
            state, amp = state_amp
            f, af = state[c]
            if af[i] == 1:
                return None
            sign = (-1) ** self.count_fermions_before(state, "af", c, i)
            new_af = af.copy()
            new_af[i] = 1
            new_state = list(state)
            new_state[c] = (f, new_af)
            return tuple(new_state), amp * sign
        return op
    
    def b(self, i, c):
        def op(state_amp):
            if state_amp is None:
                return None
            state, amp = state_amp
            f, af = state[c]
            if af[i] == 0:
                return None
            sign = (-1) ** self.count_fermions_before(state, "af", c, i)
            new_af = af.copy()
            new_af[i] = 0
            new_state = list(state)
            new_state[c] = (f, new_af)
            return tuple(new_state), amp * sign
        return op
    
    # Composes a list of operators and applies them in order
    def compose(self, *ops):
        def composed_op(state_amp):
            for op in reversed(ops):  # rightmost acts first
                state_amp = op(state_amp)
                if state_amp is None:
                    return None
            return state_amp
        return composed_op
    

    def build_gam_tau_terms(self):
        """
        Dynamically builds all operator functions and stores them in self.gam_tau_terms.   
        """

        # Build a composed four-body operator based on a symbolic spec
        def make_four_body(spec):
            """
            Build a composed four-body operator based on a symbolic spec.

            Parameters:
            - spec: list of 4 tuples (kind, dagger), where kind is "a" or "b"

            Returns:
            - A function that takes (n, m, p, q, i, j, k, l) for mode and color indices,
            and returns an operator acting on (state, amp) pairs.
            """
            def gam_tau_fn(n, m, p, q, i, j, k, l):
                ops = []
                for (kind, dagger), mode_idx, color_idx in zip(spec, (n, m, p, q), (i, j, k, l)):
                    if kind == "a":
                        ops.append(self.a_dag(mode_idx, color_idx) if dagger else self.a(mode_idx, color_idx))
                    elif kind == "b":
                        ops.append(self.b_dag(mode_idx, color_idx) if dagger else self.b(mode_idx, color_idx))
                    else:
                        raise ValueError(f"Unknown operator kind: {kind}")
                return self.compose(*ops)
            return gam_tau_fn
        
        # Build a composed two-body operator based on a symbolic spec
        def make_two_body(spec):
            """
            Build a composed two-body operator based on a symbolic spec.

            Parameters:
            - spec: list of 2 tuples (kind, dagger), where kind is "a" or "b"

            Returns:
            - A function that takes (n, q, i, l) for mode and color indices,
            and returns an operator acting on (state, amp) pairs.
            """
            def gam_tau_fn(n, q, i, l):
                ops = []
                for (kind, dagger), mode_idx, color_idx in zip(spec, (n, q), (i, l)):
                    if kind == "a":
                        ops.append(self.a_dag(mode_idx, color_idx) if dagger else self.a(mode_idx, color_idx))
                    elif kind == "b":
                        ops.append(self.b_dag(mode_idx, color_idx) if dagger else self.b(mode_idx, color_idx))
                    else:
                        raise ValueError(f"Unknown operator kind: {kind}")
                return self.compose(*ops)
            return gam_tau_fn
        
        # Finally specify symbolically the strings of operators and store them in the gam_tau_terms dictionary
        self.gam_tau_terms = {
            "g1": [make_four_body([("a", False), ("a", False), ("b", False), ("b", False)])],
            "g2": [
                make_four_body([("a", True), ("b", False), ("a", False), ("a", False)]),
                make_four_body([("b", True), ("a", False), ("b", False), ("b", False)])
            ],
            "g3": [
                make_four_body([("a", True), ("a", True), ("a", False), ("a", False)]),
                make_four_body([("b", True), ("b", True), ("b", False), ("b", False)])
            ],
            "g4a": [make_four_body([("a", True), ("b", True), ("b", False), ("a", False)])],
            "g4b": [make_four_body([("a", True), ("b", True), ("b", False), ("a", False)])],
            "g5": [make_two_body([("b", False), ("a", False)])],
            "g6": [
                make_two_body([("a", True), ("a", False)]),
                make_two_body([("b", True), ("b", False)])
            ]
        }






class NonAbelianHamiltonian():
    """
    A class to build and store the Hamiltonian terms H0 and V for the nonabelian gauge theory.
    """
    def __init__(self, Emax, L, M, Nc, B=0, n_cores=1, basis=None):
        """
        Initialize the Hamiltonian class with parameters for the gauge theory.

        Parameters:
        Emax (float): Maximum energy for the basis.
        L (float): Length of the interval.
        M (float): Mass parameter.
        Nc (int): Number of colors.
        B (int): Baryon number, default is 0.
        n_cores (int): Number of cores available to parallelize over. Default is 1.
        basis (Basis): Optional pre-built basis object.

        Outputs:
        stateList (list): List of basis states.
        size (int): Size of the basis.
        stateDict (dict): Dictionary mapping states to indices.
        """
        self.Emax = Emax
        self.L = L
        self.M = M
        self.Nc = Nc
        self.B = B
        self.n_cores = n_cores

        self.casimir = (Nc**2 - 1) / (2 * Nc)  # Casimir operator for SU(Nc)
        #self.casimir = 1.0  # For debugging purposes, set to 1.0

        if basis is not None:
            self.basis = basis
        else:
            self.basis = Basis(Emax, L, M, Nc, B)
            
        self.stateList = self.basis.stateList
        self.stateDict = self.basis.stateDict
        self.dimH  = self.basis.size
        self.fermion_energies = self.basis.fermion_energies



    def buildH0(self):
        """
        Build the H0 Hamiltonian matrix for the Interval Nonabelian Gauge Theory.

        Assumes:
        - self.stateList contains tuples of Nc items: each item is a list pair [fermion_occ, antifermion_occ]
        - self.fermion_energies is a 1D numpy array of shape (Nmodes,)

        Returns:
        scipy.sparse.csr_matrix: Sparse matrix representation of H0.
        """
        # Unpack and convert stateList into a 4D numpy array: (num_states, Nc, 2, Nmodes)
        state_array = np.array(
            [
                [np.array([f, af]) for f, af in state]
                for state in map(self.basis.unpack_state, self.stateList)
            ],
            dtype=int
        )

        # Sum over fermion and antifermion occupations: shape becomes (num_states, Nc, Nmodes)
        occ_array = state_array.sum(axis=2)

        # Now sum over color sectors: shape becomes (num_states, Nmodes)
        total_occ = occ_array.sum(axis=1)

        # Dot with fermion energies: shape (num_states,)
        energy_list = total_occ @ self.fermion_energies

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
        c1list, c2list, c3list, c4alist, c4blist, c5list, c6list = [], [], [], [], [], [], []

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
                            vala = 1 / (2 * pi ** 2 * (n + m + 1) ** 2) 
                            valb = - 1.0 / 3.0
                        elif n == l and m < k:
                            vala = 2 * self.f(n + m + 1, k + l + 1) 
                            valb = - 2 * self.f(n - l, m - k)
                        elif n < l:
                            vala = 2 * self.f(n + m + 1, k + l + 1) 
                            valb = - 2 * self.f(n - l, m - k)
                        else:
                            vala = 0
                            valb = 0
                        if vala != 0:
                            c4alist.append([n, m, k, l, vala])
                        if valb != 0:
                            c4blist.append([n, m, k, l, valb])

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

        return c1list, c2list, c3list, c4alist, c4blist, c5list, c6list
    
    
    def gencolor_coeffs(self):
        """
        Express the color tensor corresponding to 
        sum_a T^a_ij T^a_kl in a sparse format.

        Parameters:
        Nc (int): Number of colors.

        Returns:
        tuple: Lists of coefficients for the color tensor.
        """
        Nc = self.Nc
        tljki, tlkji, tikjl, tijkl, tilkj, cas = [], [], [], [], [], []

        for i in range(Nc):
            for j in range(Nc):
                for k in range(Nc):
                    for l in range(Nc):
                        val = 0.
                        if i == l and j == k:
                            val += 1/2.
                        if i == j and k == l:
                            val -= 1/(2. * Nc)
                        if val != 0:
                            tljki.append([l, j, k, i, val])
                            tlkji.append([l, k, j, i, val])
                            tikjl.append([i, k, j, l, val])
                            tijkl.append([i, j, k, l, val])
                            tilkj.append([i, l, k, j, val])
                    
        for i in range(Nc):
            for j in range(Nc):
                if i == j:
                    cas.append([i, j, self.casimir])

        return tljki, tlkji, tikjl, tijkl, tilkj, cas
    
    '''
    def gencolor_coeffs(self):
        """
        This is for debugging!

        Express the color tensor corresponding to 
        sum_a T^a_ij T^a_kl in a sparse format.

        Parameters:
        Nc (int): Number of colors.

        Returns:
        tuple: Lists of coefficients for the color tensor.
        """
        Nc = self.Nc
        tikjl, tijkl, tilkj, cas = [], [], [], []

        for i in range(Nc):
            for j in range(Nc):
                for k in range(Nc):
                    for l in range(Nc):
                        val = 0
                        if i == j == k == l:
                            val = 1
                        if val != 0:
                            tikjl.append([i, k, j, l, val])
                            tijkl.append([i, j, k, l, val])
                            tilkj.append([i, l, k, j, val])
                    
        for i in range(Nc):
            for j in range(Nc):
                if i == j:
                    cas.append([i, j, self.casimir])

        return tikjl, tijkl, tilkj, cas
    '''

    def apply_tensor_operator(self, state, gamxlist, colxlist, operator_fns, energy_check=False, tol=1e-12):
        """
        Generic routine to apply one or more tensor-weighted operators to a state.

        Parameters:
            state: ([f_list1, af_list1], ..., [f_listNc, af_listNc]), amplitude
            gamxlist: list of [n, m, p, q, coeff] or [p, q, coeff]
            colxlist: list of [i, j, k, l, coeff] or [i, l, coeff]
            operator_fns: a single function or list of functions
            energy_check: bool — whether to truncate based on energy
            tol: numerical tolerance for filtering

        Returns:
            List of (([f_list1, af_list1], ..., [f_listNc, af_listNc]), amplitude) states after action
        """ 
        result_dict = defaultdict(float)

        if isinstance(operator_fns, Callable):
            operator_fns = [operator_fns]

        for gcoeff_entry in gamxlist:
            *gam_indices, gcoeff = gcoeff_entry
            for ccoeff_entry in colxlist:
                *col_indices, ccoeff = ccoeff_entry

                for op_fn in operator_fns:
                    op = op_fn(*gam_indices, *col_indices)
                    new_state = op(state)
                    if new_state is None:
                        continue

                    (new_color_state, amp) = new_state
                    amp *= gcoeff * ccoeff

                    if abs(amp) < tol:
                        continue

                    if energy_check:
                        energy = self.basis.stateEnergy(new_color_state)
                        if energy > self.Emax:
                            continue

                    key = tuple((tuple(f), tuple(af)) for f, af in new_color_state)
                    result_dict[key] += amp

        return [(([(list(f), list(af)) for f, af in key]), amp)
            for key, amp in result_dict.items() if abs(amp) > tol]


    def apply_all_terms(
        self,
        state,
        coeff_lists,
        operator_sets,
        energy_flags,
        tol=1e-12
    ):
        """
        Apply all interaction terms to a single nonabelian state and return the result.

        Parameters:
            state: ([ [f_list1, af_list1], ..., [f_listNc, af_listNc] ], amplitude)
            coeff_lists: dict of {gn_label: (gamxlist, colxlist)}
            operator_sets: dict of {gn_label: operator_generator or list thereof}
            energy_flags: dict of {gn_label: bool}
            tol: numerical tolerance

        Returns:
            List of ( [ [f_list1, af_list1], ..., [f_listNc, af_listNc] ], amplitude ) states
        """
        result_dict = defaultdict(float)

        for label, (gamxlist, colxlist) in coeff_lists.items():
            op_fns = operator_sets[label]
            energy_check = energy_flags.get(label, False)

            term_results = self.apply_tensor_operator(
                state,
                gamxlist,
                colxlist,
                op_fns,
                energy_check=energy_check,
                tol=tol
            )

            for (color_state, amp) in term_results:
                key = tuple((tuple(f), tuple(af)) for f, af in color_state)
                result_dict[key] += amp

        return [([ [list(f), list(af)] for f, af in key ], amp)
                    for key, amp in result_dict.items()
                    if abs(amp) > tol]

    
    def process_state_column_static(
        self,
        col_index,
        packed,
        basis,
        coeff_lists,
        operator_sets,
        energy_check_flags,
        Emax,
        tol
    ):
        """
        Process one basis state at column `col_index` and return all nonzero matrix elements.

        Parameters:
            col_index: integer column index
            packed: packed representation of the state
            basis: basis object (with .unpack_state and .stateEnergy)
            coeff_lists: dict of {label: (gamxlist, colxlist)}
            operator_sets: dict of {label: operator function(s)}
            energy_check_flags: dict of {label: bool}
            Emax: energy cutoff
            tol: amplitude cutoff

        Returns:
            List of (tupled_state, col_index, amplitude)
        """
        unpacked_state = basis.unpack_state(packed)  # → [[f1, af1], ..., [fNc, afNc]]
        state = (unpacked_state, 1.0)
        results = []

        for label, (gamxlist, colxlist) in coeff_lists.items():
            op_fns = operator_sets[label]
            energy_check = energy_check_flags.get(label, False)

            if isinstance(op_fns, Callable):
                op_fns = [op_fns]

            for gam_entry in gamxlist:
                *gam_indices, gam_coeff = gam_entry

                for col_entry in colxlist:
                    *col_indices, col_coeff = col_entry

                    for op_fn in op_fns:
                        op = op_fn(*gam_indices, *col_indices)
                        new_state = op(state)

                        if new_state is None:
                            continue

                        new_color_state, amp = new_state
                        amp *= gam_coeff * col_coeff

                        if abs(amp) < tol:
                            continue

                        if energy_check:
                            energy = basis.stateEnergy(new_color_state)
                            if energy > Emax:
                                continue

                        key = tuple([list(f), list(af)] for f, af in new_color_state)
                        results.append((key, col_index, amp))

        return results


    def buildV(self, start=0, end=None):
        """
        Build the V Hamiltonian matrix for the nonabelian theory over a slice of basis states.

        Parameters:
            start (int): Starting index in basis.stateList (inclusive).
            end (int): Ending index in basis.stateList (exclusive). If None, goes to end.

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix representation of V slice.
        """
        # Instantiate operator sets and interaction coefficients
        ops = NonAbelianFermionOperators(self.Nc)
        gam1, gam2, gam3, gam4a, gam4b, gam5, gam6 = self.gencoeffs()  # These are the gamxlists
        tljki, tlkji, tikjl, tijkl, tilkj, casimir               = self.gencolor_coeffs()  # These are the colxlists

        coeff_lists = {
            "g1": (gam1, tljki),
            "g2": (gam2, tlkji),
            "g3": (gam3, tikjl),
            "g4a": (gam4a, tijkl),
            "g4b": (gam4b, tilkj),
            "g5": (gam5, casimir),
            "g6": (gam6, casimir),
        }
    
        operator_sets = ops.gam_tau_terms

        energy_check_flags = {
            "g1": False,
            "g2": True,
            "g3": True,
            "g4a": True,
            "g4b": True,
            "g5": False,
            "g6": True,
        }

        if end is None:
            end = len(self.basis.stateList)

        # Prepare arguments for parallel processing
        args = [
            (i, packed, self.basis, coeff_lists, operator_sets, energy_check_flags, self.Emax, 1e-12)
            for i, packed in enumerate(self.basis.stateList[start:end], start)
        ]

        # Parallel processing of columns
        results = Parallel(n_jobs=self.n_cores)(
            delayed(self.process_state_column_static)(*arg)
            for arg in args
        )

        # Flatten results and build sparse matrix entries
        rows, cols, data = [], [], []
        for termlist in results:
            for key_state, col, amp in termlist: #type: ignore
                # Convert key_state back to list-of-lists to match pack_state input
                unpacked = [list(faf) for faf in key_state]  # faf = [f, af]
                packed_row = self.basis.pack_state(unpacked)
                row_index = self.basis.stateDict.get(packed_row, -1)
                if row_index != -1:
                    rows.append(row_index)
                    cols.append(col)
                    data.append(amp)

        dim = self.dimH
        self.V = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))

