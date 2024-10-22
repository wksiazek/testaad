from DET.DETAlgs.base import BaseAlg
from DET.DETAlgs.data.alg_data import IDEData
from DET.DETAlgs.methods.methods_de import selection, mutation
from DET.DETAlgs.methods.methods_ide import ide_get_f, ide_get_cr, ide_binomial_crossing
from DET.models.enums.boundary_constrain import fix_boundary_constraints

"""
    IDE

    Links:
    https://www.scirp.org/journal/paperinformation?paperid=96749

    References:
    Ma, J. and Li, H. (2019) Research on Rosenbrock Function Optimization Problem Based on Improved Differential 
    Evolution Algorithm. 
    Journal of Computer and Communications, 7, 107-120. doi: 10.4236/jcc.2019.711008. 
"""


class IDE(BaseAlg):
    def __init__(self, params: IDEData, db_conn=None, db_auto_write=False):
        super().__init__(IDE.__name__, params, db_conn, db_auto_write)

    def next_epoch(self):
        # Calculate F and CR
        f = ide_get_f(self._epoch_number, self.num_of_epochs)
        cr_arr = ide_get_cr(self._pop)

        # New population after mutation
        v_pop = mutation(self._pop, f)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = ide_binomial_crossing(self._pop, v_pop, cr_arr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
