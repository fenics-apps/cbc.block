from __future__ import division

"""Block operations for linear algebra.

To make this work, all operators should define at least a __mul__(self, other)
method, which either does its thing (typically if isinstance(other,
BlockVector)), or returns a BlockCompose(self, other) object which defers the
action until there is a BlockVector to work on.

In addition, methods are injected into dolfin.Matrix / dolfin.Vector as needed.

NOTE: Nested blocks SHOULD work but has not been tested.
"""

import dolfin
from block_mat import block_mat
from block_vec import block_vec
from block_compose import block_compose
from block_bc import block_bc

# To make stuff like L=C*B work when C and B are type dolfin.Matrix, we inject
# methods into dolfin.Matrix

_old_mat_mul = dolfin.Matrix.__mul__
def _mat_mul(self, x):
    y = _old_mat_mul(self, x)
    if y == NotImplemented:
        y = block_compose(self, x)
    return y
dolfin.Matrix.__mul__  = _mat_mul
del _mat_mul

dolfin.Matrix.__rmul__ = lambda self, other: block_compose(other, self)
dolfin.Matrix.__neg__  = lambda self       : block_compose(-1, self)

# For the Trilinos stuff, it's much nicer if down_cast is a method on the object
dolfin.Matrix.down_cast        = dolfin.down_cast
dolfin.GenericMatrix.down_cast = dolfin.down_cast
dolfin.Vector.down_cast        = dolfin.down_cast
dolfin.GenericVector.down_cast = dolfin.down_cast
