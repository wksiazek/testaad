from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import DERLData
from DET.DETAlgs.methods.methods_de import binomial_crossing, selection
from DET.DETAlgs.methods.methods_derl import derl_mutation
from DET.models.enums.boundary_constrain import fix_boundary_constraints

"""
    DERL

    Links:
    https://www.sciencedirect.com/science/article/pii/S037722170500281X

    References:
    Kaelo, P. & Ali, Montaz. (2006). A numerical study of some modified differential evolution algorithms. 
    European Journal of Operational Research. 169. 1176-1184. 10.1016/j.ejor.2004.08.047. 
"""


class DERL(BaseAlg):
    def __init__(self, params: DERLData, db_conn=None, db_auto_write=False):
        super().__init__(DERL.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr

    def next_epoch(self):
        # New population after mutation
        v_pop = derl_mutation(self._pop)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
