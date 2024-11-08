from __future__ import division
from __future__ import absolute_import
from builtins import map
from builtins import range
from . import *
from .block_util import block_tensor, isscalar, wrap_in_list, create_vec_from

def block_assemble(lhs, rhs=None, bcs=None,
                   symmetric=False, signs=None, symmetric_mod=None):
    """
    Assembles block matrices, block vectors or block systems.
    Input can be arrays of variational forms or block matrices/vectors.

    Arguments:

            symmetric : Boundary conditions are applied so that symmetry of the system
                        is preserved. If only the left hand side of the system is given,
                        then a matrix represententing the rhs corrections is returned
                        along with a symmetric matrix. NOTE: Symmetric assembly not
                        supported in parallel (MPI).

        symmetric_mod : Matrix describing symmetric corrections for assembly of the
                        of the rhs of a variational system.

                signs : An array to specify the signs of diagonal blocks. The sign
                        of the blocks are computed if the argument is not provided.
    """
    error_msg = {'incompatibility' : 'A and b do not have compatible dimensions.',
                 'symm_mod error'  : 'symmetric_mod argument only accepted when assembling a vector',
                 'not square'      : 'A must be square for symmetric assembling',
                 'invalid bcs'     : 'Expecting a list or list of lists of DirichletBC.',
                 'invalid signs'   : 'signs should be a list of length n containing only 1 or -1'}
    # Check arguments
    from numpy import ndarray
    has_rhs = True if isinstance(rhs, ndarray) else rhs != None
    has_lhs = True if isinstance(rhs, ndarray) else rhs != None

    if has_lhs and has_rhs:
        A, b = list(map(block_tensor,[lhs,rhs]))
        n, m = A.blocks.shape
        if not ( isinstance(b,block_vec) and  len(b.blocks) is m):
            raise TypeError(error_msg['incompatibility'])
    else:
        A, b = block_tensor(lhs), None
        if isinstance(A,block_vec):
            A, b = None, A
            n, m = 0, len(b.blocks)
        else:
            n,m = A.blocks.shape
    if A and symmetric and (m is not n):
        raise RuntimeError(error_msg['not square'])
    if symmetric_mod and ( A or not b ):
        raise RuntimeError(error_msg['symmetric_mod error'])
    # First assemble everything needing assembling.
    from dolfin import assemble
    assemble_if_form = lambda x: assemble(x, keep_diagonal=True) if _is_form(x) else x
    if A:
        A.blocks.flat[:] = list(map(assemble_if_form,A.blocks.flat))
    if b:
        #b.blocks.flat[:] = map(assemble_if_form, b.blocks.flat)
        b = block_vec(list(map(assemble_if_form, b.blocks.flat)))
    # If there are no boundary conditions then we are done.
    if bcs is None:
        return [A,b] if (A and b) else A or b

    # Apply supplied RHS BCs if we are only assembling the right hand side. If we are
    # assembling A anyway, we use that for symmetry preservation instead.
    if A is None and symmetric_mod:
        symmetric_mod.apply(b)
        return b

    # check if arguments are forms, in which case bcs have to be split
    from ufl_legacy import Form
    lhs_bcs = (block_bc.from_mixed if isinstance(lhs, Form) else block_bc)(bcs, symmetric=symmetric, signs=signs)

    result = []
    if A:
        rhs_bcs = lhs_bcs.apply(A)
        result.append(A)
    else:
        rhs_bcs = lhs_bcs.rhs(None)
    if symmetric and A:
        result.append(rhs_bcs)
    if b:
        rhs_bcs.apply(b)
        result.append(b)
    return result[0] if len(result)==1 else result


def block_symmetric_assemble(forms, bcs):
    return block_assemble(forms,bcs=bcs,symmetric=True)

def _is_form(form):
    from dolfin import Form as cpp_Form
    from ufl_legacy.form import Form as ufl_Form
    return isinstance(form, (cpp_Form, ufl_Form))

def _new_square_matrix(bc, val):
    from dolfin import TrialFunction, TestFunction, FunctionSpace
    from dolfin import assemble, Constant, inner, dx
    import numpy
    
    V = FunctionSpace(bc.function_space())
    
    u,v = TrialFunction(V),TestFunction(V)
    Z = assemble(Constant(0)*inner(u,v)*dx)
    if val != 0.0:
        lrange = list(range(*Z.local_range(0)))
        idx = numpy.ndarray(len(lrange), dtype=numpy.intc)
        idx[:] = lrange
        Z.ident(idx)
        if val != 1.0:
            Z *= val
    return Z
