from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import OppBasedData
from DET.DETAlgs.methods.methods_opposition_based import opp_based_generation_jumping
from DET.DETAlgs.methods.methods_de import mutation, binomial_crossing, selection
from DET.models.enums.boundary_constrain import fix_boundary_constraints

"""
    OppBasedDE

    Links:
    https://ieeexplore.ieee.org/document/4358759

    References:
    S. Rahnamayan, H. R. Tizhoosh and M. M. A. Salama, "Opposition-Based Differential Evolution," 
    in IEEE Transactions on Evolutionary Computation, vol. 12, no. 1, pp. 64-79, Feb. 2008, 
    doi: 10.1109/TEVC.2007.894200.
"""


class OppBasedDE(BaseAlg):
    def __init__(self, params: OppBasedData, db_conn=None, db_auto_write=False):
        super().__init__(OppBasedDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.nfc = 0  # number of function calls
        self.max_nfc = params.max_nfc
        self.jumping_rate = params.jumping_rate

    def next_epoch(self):
        # New population after mutation
        v_pop = mutation(self._pop, f=self.mutation_factor)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)
        self.nfc += self.population_size

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Generation jumping
        if opp_based_generation_jumping(new_pop, self.jumping_rate, self._function.eval):
            self.nfc += self.population_size

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
