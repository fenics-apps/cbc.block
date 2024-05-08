from block import *
from block.algebraic.hazmath import AMG as hazAMG
from block.algebraic.petsc import KSP, AMG
from block.dolfin_util import BoxBoundary
from block.iterative import Richardson
from dolfin import *

#########################
# Basic Poisson (poisson.py)
#########################

mesh = UnitSquareMesh(32,32)
V = FunctionSpace(mesh, "CG", 1)

f = Expression("sin(pi*x[0])", degree=2)
u, v = TrialFunction(V), TestFunction(V)

a = u*v*dx + dot(grad(u), grad(v))*dx
L = f*v*dx

A = assemble(a)
b = assemble(L)

u = Function(V).vector()
solve(A, u, b)

#########################
# Solve as single-matrix
#########################

Apre = AMG(A)
Ainv = KSP(A, ksp_type='cg', pc_type='ilu')
# Alternatively:
#Ainv = KSP(A, ksp_type='cg', pc_type='hypre', pc_hypre_type='boomeramg')
x = Ainv*b

check_expected('x', x, expected=u, show=True)

#########################
# Solve a (trivial) block-matrix, using KSP Ainv as preconditioner
#########################

AA = block_mat([[A]])
bb = block_vec([b])
AApre = block_mat([[Ainv]])
AAinv = KSP(AA, precond=AApre, ksp_type='cg')

xx = AAinv*bb

check_expected('x', xx[0], expected=u, show=True)

#########################
# Solve a nontrivial block-matrix (mixedpoisson.py)
#########################

BDM = FunctionSpace(mesh, "BDM", 1)
DG = FunctionSpace(mesh, "DG", 0)

tau, sigma = TestFunction(BDM), TrialFunction(BDM)
v,   u     = TestFunction(DG),  TrialFunction(DG)

class BoundarySource(UserExpression):
    def __init__(self, mesh):
        super().__init__(self)
        self.mesh = mesh
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = sin(5*x[0])
        values[0] = g*n[0]
        values[1] = g*n[1]
    def value_shape(self):
        return (2,)

G = BoundarySource(mesh)
boundary = BoxBoundary(mesh).ns
bcs_BDM = [DirichletBC(BDM, G, boundary)]
bcs = block_bc([bcs_BDM, None], symmetric=True)

f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=4)

a11 = dot(sigma, tau) * dx
a12 = div(tau) * u * dx
a21 = div(sigma) * v *dx
L2  = - f * v * dx

AA = block_assemble([[a11, a12],
                     [a21,  0 ]])

bb = block_assemble([0, L2])

bcs.apply(AA).apply(bb)

# Extract the individual submatrices
[[A, B],
 [C, _]] = AA

Ap = hazAMG(A)
Lp = Richardson(C*B, precond=0.5, iter=40, name='L^', show=0)

AAp = block_mat([[Ap, 0],
                 [0,  Lp]])

AAinv = KSP(AA, precond=AAp, ksp_type='minres')
Sigma, U = AAinv * bb

print('expected (from mixedpoisson.py): Sigma=1.2138, U=6.7166')
check_expected('Sigma', Sigma, show=True)
check_expected('U', U, show=True)
