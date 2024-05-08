from __future__ import division

from builtins import str
from block.block_base import block_base
from block.helpers import supports_mpi
from block.object_pool import vec_pool
from petsc4py import PETSc
import dolfin as df
import numpy as np
import hashlib
from time import time
from dolfin import info

class petsc_base(block_base):
    @classmethod
    def _merge_options(cls, prefix, options, defaults):
        # The PETSc options database is global, so we create a unique prefix
        # that precisely matches the supplied options. The user can override
        # this prefix if necessary.
        if prefix is None:
            # Use class name instead of hash of defaults (we assume defaults
            # are constant for a given class)
            prefix = cls.__name__
            if options:
                sha = hashlib.sha256(str(tuple(sorted(options.items()))).encode())
                prefix += sha.hexdigest()[:8]
            prefix += ':'

        optsDB = PETSc.Options(prefix)
        params = defaults.copy()
        if options:
            params.update(options)
        for key, val in params.items():
            # Allow overriding defaults by setting key='del'. Command-line
            # options (already in optsDB) are never overwritten.
            if val!='del' and not optsDB.hasName(key):
                optsDB.setValue(key, val)
        return prefix, optsDB

    def down_cast(self):
        return self.petsc_op

class precond(petsc_base):
    def __init__(self, A, prectype, pdes, nullspace, prefix, options, V=None, defaults={}):
        Ad = mat(A)

        if nullspace:
            from block.block_util import isscalar
            ns = PETSc.NullSpace()
            if isscalar(nullspace):
                ns.create(constant=True)
            else:
                ns.create(constant=False, vectors=[vec(v) for v in nullspace])
            try:
                Ad.setNearNullSpace(ns)
            except:
                info('failed to set near null space (not supported in petsc4py version)')

        prefix, self.optsDB = self._merge_options(prefix=prefix, options=options, defaults=defaults)

        if prectype == PETSc.PC.Type.HYPRE and nullspace:
            if not all(self.optsDB.hasName(n) for n in ['pc_hypre_boomeramg_nodal_coarsen', 'pc_hypre_boomeramg_vec_interp_variant']):
                print('Near null space is ignored by hypre UNLESS you also supply pc_hypre_boomeramg_nodal_coarsen and pc_hypre_boomeramg_vec_interp_variant')

        self.A = A
        self.petsc_op = PETSc.PC().create(V.mesh().mpi_comm() if V else None)
        self.petsc_op.setOptionsPrefix(prefix)

        if prectype:
            try:
                self.petsc_op.setType(prectype)
            except PETSc.Error as e:
                if e.ierr == 86:
                    raise ValueError(f'Unknown PETSc type "{prectype}"')
                else:
                    raise

            self.petsc_op.setOperators(Ad)
            self.petsc_op.setFromOptions()

        self.setup_time = -1.

    def matvec(self, b):
        # NOTE: we want to be lazy only setup when the action is needed
        if self.setup_time < 0:
            T = time()
            # Create preconditioner based on the options database
            self.petsc_op.setUp()

            setup_time = time()-T
            comm = self.petsc_op.getComm().tompi4py()
            
            setup_time = comm.allreduce(setup_time)/comm.size
            info('constructed %s preconditioner in %.2f s'%(self.__class__.__name__, setup_time))

            self.setup_time = setup_time
            
        from dolfin import GenericVector, PETScVector
        if not isinstance(b, (PETScVector, GenericVector)):
            return NotImplemented
        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for PETSc matvec, %d != %d'%(len(x),len(b)))

        self.petsc_op.apply(vec(b), vec(x))
        return x

    def __str__(self):
        return '<%s prec of %s>'%(self.__class__.__name__, str(self.A))

    def create_vec(self, dim=1):
        return self.A.create_vec(dim)

class ML(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None, prefix=None):
        super().__init__(A, PETSc.PC.Type.ML,
                         pdes=pdes, nullspace=nullspace, options=parameters, prefix=prefix,
                         defaults={
                             # Symmetry- and PD-preserving smoother
                             'mg_levels_ksp_type': 'chebyshev',
                             'mg_levels_pc_type':  'jacobi',
                             # Fixed number of iterations to preserve linearity
                             'mg_levels_ksp_max_it':               2,
                             'mg_levels_ksp_check_norm_iteration': 9999,
                             # Exact inverse on coarse grid
                             'mg_coarse_ksp_type': 'preonly',
                             'mg_coarse_pc_type':  'lu',
                         })

class ILU(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None, prefix=None):
        supports_mpi(False, 'PETSc ILU does not work in parallel', mat(A).comm.size)
        super().__init__(A, PETSc.PC.Type.ILU,
                         pdes=pdes, nullspace=nullspace, options=parameters, prefix=prefix)

class Cholesky(precond):
    def __init__(self, A, parameters=None, prefix=None):
        super().__init__(A, PETSc.PC.Type.CHOLESKY, 1, None, options=parameters, prefix=prefix)

class LU(precond):
    def __init__(self, A, parameters=None, prefix=None):
        super().__init__(A, PETSc.PC.Type.CHOLESKY,
                         pdes=1, nullspace=None, options=parameters, prefix=prefix)

class LU(precond):
    def __init__(self, A, parameters=None, prefix=None, defaults={}):
        super().__init__(A, PETSc.PC.Type.LU,
                         pdes=1, nullspace=None, options=parameters, prefix=prefix,
                         defaults=defaults)


class UMFPACK_LU(LU):
    def __init__(self, A, parameters=None, prefix=None):
        super().__init__(A,
                         parameters=parameters, prefix=prefix,
                         defaults={
                             'pc_factor_mat_solver_type': 'umfpack',
                         })

        
class MUMPS_LU(LU):
    def __init__(self, A, parameters=None, prefix=None):
        super().__init__(A,
                         parameters=parameters, prefix=prefix,
                         defaults={
                             'pc_factor_mat_solver_type': 'mumps',
                         })


class SUPERLU_LU(LU):
    def __init__(self, A, parameters=None, prefix=None):
        super().__init__(A,
                         parameters=parameters, prefix=prefix,
                         defaults={
                             'pc_factor_mat_solver_type': 'superlu',
                         })
        self.petsc_op.setFactorPivot(1E-16)

class AMG(precond):
    """
    BoomerAMG preconditioner from the Hypre Library
    """
    def __init__(self, A, parameters=None, pdes=1, nullspace=None, prefix=None):
        super().__init__(A, PETSc.PC.Type.HYPRE,
                         pdes=pdes, nullspace=nullspace, options=parameters, prefix=prefix,
                         defaults={
                             "pc_hypre_type": "boomeramg",
                             #"pc_hypre_boomeramg_cycle_type": "V", # (V,W)
                             #"pc_hypre_boomeramg_max_levels": 25,
                             #"pc_hypre_boomeramg_max_iter": 1,
                             #"pc_hypre_boomeramg_tol": 0,     
                             #"pc_hypre_boomeramg_truncfactor" : 0,      # Truncation factor for interpolation
                             #"pc_hypre_boomeramg_P_max": 0,             # Max elements per row for interpolation
                             #"pc_hypre_boomeramg_agg_nl": 0,            # Number of levels of aggressive coarsening
                             #"pc_hypre_boomeramg_agg_num_paths": 1,     # Number of paths for aggressive coarsening
                             #"pc_hypre_boomeramg_strong_threshold": .25,# Threshold for being strongly connected
                             #"pc_hypre_boomeramg_max_row_sum": 0.9,
                             #"pc_hypre_boomeramg_grid_sweeps_all": 1,   # Number of sweeps for the up and down grid levels 
                             #"pc_hypre_boomeramg_grid_sweeps_down": 1,  
                             #"pc_hypre_boomeramg_grid_sweeps_up":1,
                             #"pc_hypre_boomeramg_grid_sweeps_coarse": 1,# Number of sweeps for the coarse level (None)
                             #"pc_hypre_boomeramg_relax_type_all":  "symmetric-SOR/Jacobi", # (Jacobi, sequential-Gauss-Seidel, seqboundary-Gauss-Seidel, 
                                                                                            #  SOR/Jacobi, backward-SOR/Jacobi,  symmetric-SOR/Jacobi,  
                                                                                            #  l1scaled-SOR/Jacobi Gaussian-elimination, CG, Chebyshev, 
                                                                                            #  FCF-Jacobi, l1scaled-Jacobi)
                             #"pc_hypre_boomeramg_relax_type_down": "symmetric-SOR/Jacobi",
                             #"pc_hypre_boomeramg_relax_type_up": "symmetric-SOR/Jacobi",
                             #"pc_hypre_boomeramg_relax_type_coarse": "Gaussian-elimination",
                             #"pc_hypre_boomeramg_relax_weight_all": 1,   # Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps)
                             #"pc_hypre_boomeramg_relax_weight_level": (1,1), # Set the relaxation weight for a particular level
                             #"pc_hypre_boomeramg_outer_relax_weight_all": 1,
                             #"pc_hypre_boomeramg_outer_relax_weight_level": (1,1),
                             #"pc_hypre_boomeramg_no_CF": "",               # Do not use CF-relaxation 
                             #"pc_hypre_boomeramg_measure_type": "local",   # (local global)
                             #"pc_hypre_boomeramg_coarsen_type": "Falgout", # (Ruge-Stueben, modifiedRuge-Stueben, Falgout, PMIS, HMIS)
                             #"pc_hypre_boomeramg_interp_type": "classical",# (classical, direct, multipass, multipass-wts, ext+i, ext+i-cc, standard, standard-wts, FF, FF1)
                             #"pc_hypre_boomeramg_print_statistics": "",
                             #"pc_hypre_boomeramg_print_debug": "",
                             #"pc_hypre_boomeramg_nodal_coarsen": "",
                             #"pc_hypre_boomeramg_nodal_relaxation": "",
                         })


class SOR(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None, prefix=None):
        options = {
            }
        super().__init__(A, PETSc.PC.Type.SOR,
                         pdes=pdes, nullspace=nullspace, options=parameters, prefix=prefix,
                         defaults={
                             "pc_sor_omega": 1,      # relaxation factor (0 < omega < 2, 1 is Gauss-Seidel)
                             "pc_sor_its": 1,        # number of inner SOR iterations
                             "pc_sor_lits": 1,       # number of local inner SOR iterations
                             "pc_sor_symmetric": "", # for SSOR
                             #"pc_sor_backward": "",
                             #"pc_sor_forward": "",  
                             #"tmp_pc_sor_local_symmetric": "", # use SSOR separately on each processor
                             #"tmp_pc_sor_local_backward": "",  
                             #"tmp_pc_sor_local_forward": "",
                         })
        supports_mpi(not self.optsDB.hasName('pc_sor_symmetric'), 'PETSc symmetric SOR not supported in parallel', mat(A).comm.size)


class Elasticity(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None, prefix=None):
        super().__init__(A, PETSc.PC.Type.GAMG,
                         pdes=pdes, nullspace=nullspace, options=parameters, prefix=prefix,
                         defaults={
                             'pc_mg_cycle_type': 'w',
                             'pc_mg_multiplicative_cycles': 2
                         })

class ASM(precond):
    """
    Additive Scwharz Method.
    Defaults (or should default, not tested) to one block per process.
    """
    def __init__(self, A, parameters=None, pdes=1, nullspace=None, prefix=None):
        super().__init__(A, PETSc.PC.Type.ASM,
                         pdes=pdes, nullspace=nullspace, options=parameters, prefix=prefix,
                         defaults={
                             #"pc_asm_blocks":  1,             # Number of subdomains
                             "pc_asm_overlap": 1,             # Number of grid points overlap
                             "pc_asm_type":  "RESTRICT",      # (NONE, RESTRICT, INTERPOLATE, BASIC)
                             "sup_ksp_type": "preonly",       # KSP solver for the subproblems
                             "sub_pc_type": "ilu"             # Preconditioner for the subproblems
                         })


class Jacobi(precond):
    """
    Actually this is only a diagonal scaling preconditioner; no support for relaxation or multiple iterations.
    """
    def __init__(self, A, parameters=None, pdes=1, nullspace=None, prefix=None):
        super().__init__(A, PETSc.PC.Type.JACOBI,
                         pdes=pdes, nullspace=nullspace, options=parameters, prefix=prefix,
                         defaults={
                             #"pc_jacobi_rowmax": "",  # Use row maximums for diagonal
                             #"pc_jacobi_rowsum": "",  # Use row sums for diagonal
                             #"pc_jacobi_abs":, "",    # Use absolute values of diagaonal entries
                         })


def vec(x):
    return df.as_backend_type(x).vec()


def mat(A):
    return df.as_backend_type(A).mat()


class HypreAMS(precond):
    '''
    AMG auxiliary space preconditioner for H(curl) problems in 2d/3d and
    H(div) problems in 2d. The bilinear form behind the matrix whose 
    approximate inverse is constructed is: Find u in V

       alpha*inner(curl(u), curl(v)) + beta*inner(u, v)*dx for all v in V

    where alpha > 0 and beta >= 0.
    '''
    def __init__(self, A, V, ams_zero_beta_poisson=False, prefix=None):
        mesh = V.mesh()
        if mesh.geometry().dim() == 2:
            assert V.ufl_element().family() in ('Raviart-Thomas', 'Nedelec 1st kind H(curl)')
        else:
            assert V.ufl_element().family() == 'Nedelec 1st kind H(curl)'
        # FIXME: only tested this for lower orders
        assert V.ufl_element().degree() == 1

        # AMS setup requires auxiliary operators
        Q = df.FunctionSpace(mesh, 'CG', 1)
        G = df.DiscreteOperators.build_gradient(V, Q)

        super().__init__(A, PETSc.PC.Type.HYPRE, V=V,
                         pdes=None, nullspace=None, options=None, prefix=prefix,
                         defaults={
                             'pc_hypre_type': 'ams',
                         })
        self.petsc_op.setHYPREDiscreteGradient(mat(G))

        # Constants
        constants = [vec(df.interpolate(df.Constant(c), V).vector())
                     for c in np.eye(mesh.geometry().dim())]
        self.petsc_op.setHYPRESetEdgeConstantVectors(*constants)

        # NOTE: signal that we don't have lower order term
        ams_zero_beta_poisson and self.petsc_op.setHYPRESetBetaPoissonMatrix(None)

        self.setup_time = -1

        
class HypreADS(precond):
    '''
    AMG auxiliary space preconditioner for H(div) problems in 3d. The bilinear 
    form behind the matrix whose approximate inverse is constructed is: Find u in V

       alpha*inner(div(u), div(v)) + beta*inner(u, v)*dx for all v in V

    where alpha > 0 and beta >= 0.
    '''
    @staticmethod
    def coordinates(mesh):
        '''As P1 functions'''
        V = df.FunctionSpace(mesh, 'CG', 1)
        
        return V.tabulate_dof_coordinates()

    @staticmethod
    def discrete_gradient(mesh):
        '''P1 to Ned1 map'''
        Ned = df.FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
        P1 = df.FunctionSpace(mesh, 'CG', 1)
    
        G = df.DiscreteOperators.build_gradient(Ned, P1)

        return mat(G)
    
    @staticmethod
    def discrete_curl(mesh):
        '''Ned1 to RT1 map'''
        assert mesh.geometry().dim() == 3
        assert mesh.topology().dim() == 3

        Ned = df.FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
        RT = df.FunctionSpace(mesh, 'RT', 1)

        RT_f2dof = np.array(RT.dofmap().entity_dofs(mesh, 2))
        Ned_e2dof = np.array(Ned.dofmap().entity_dofs(mesh, 1))

        # Alloc sparsity
        a = df.Constant(0)*df.inner(df.TestFunction(RT), df.TrialFunction(Ned))*df.dx
        C = df.PETScMatrix()
        df.assemble(a, tensor=C)

        RT_f2dof = np.array(RT.dofmap().entity_dofs(mesh, 2))
        Ned_e2dof = np.array(Ned.dofmap().entity_dofs(mesh, 1))

        # Facets in terms of edges
        mesh.init(2, 1)
        f2e = mesh.topology()(2, 1)

        row_cols, row_values = np.zeros(3, dtype='int32'), np.array([-2, 2, -2])
        Cmat = C.mat()

        for facet, row in enumerate(RT_f2dof):
            row_cols[:] = Ned_e2dof[f2e(facet)]
            Cmat.setValues([row], row_cols, row_values, PETSc.InsertMode.INSERT_VALUES)
        Cmat.assemble()

        return Cmat

    def __init__(self, A, V, prefix=None):
        assert V.mesh().geometry().dim() == 3
        assert V.ufl_element().family() == 'Raviart-Thomas'
        assert V.ufl_element().degree() == 1
        supports_mpi(False, 'ADS curl matrix needs to be fixed for parallel', mat(A).comm.size) # FIXME

        mesh = V.mesh()
        C = HypreADS.discrete_curl(mesh)
        G = HypreADS.discrete_gradient(mesh)
        xyz = HypreADS.coordinates(mesh)

        super().__init__(A, '', V=V,
                         pdes=None, nullspace=None, options=None, prefix=prefix)
        self.petsc_op.setType('hypre')        
        self.petsc_op.setHYPREType('ads')
        self.petsc_op.setOperators(mat(A))
        # Attach auxiliary operators
        self.petsc_op.setHYPREDiscreteGradient(G)
        self.petsc_op.setHYPREDiscreteCurl(G)
        self.petsc_op.setCoordinates(xyz)
        self.petsc_op.setUp()
        
