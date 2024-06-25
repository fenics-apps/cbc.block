def isequal(op1, op2, eps=1e-3):
    from . import block_vec
    v = op1.create_vec()
    block_vec([v]).randomize()
    xv = op1*v
    err = (xv-op2*v).norm('l2')/(xv).norm('l2')
    return err < eps

def issymmetric(op):
    x = op.create_vec()
    if hasattr(x, 'randomize'):
        x.randomize()
    else:
        from . import block_vec
        block_vec([x]).randomize();
    opx = op*x
    err = (opx - op.T*x).norm('l2')/opx.norm('l2')
    return (err < 1e-6)

def sign_of(op):
    from numpy.random import random
    if isscalar(op):
        return -1 if op < 0 else 1
    else:
        x = op.create_vec(dim=1)
        x.set_local(random(x.local_size()))
        x.apply('')
        return -1 if x.inner(op*x) < 0 else 1

def mult(op, x, transposed=False):
    if not transposed or isscalar(op):
        return op*x
    else:
        return op.transpmult(x)

def isscalar(obj):
    """Return True if obj is convertible to float. Use this instead of
    numpy.isscalar, becuase the latter returns true for e.g. strings"""
    try:
        float(obj)
        return True
    except:
        return False

def copy(obj):
    """Return a deep copy of the object"""
    if hasattr(obj, 'copy'):
        return obj.copy()
    else:
        import copy
        try:
            return copy.deepcopy(obj)
        except TypeError:
#            from dolfin import warning
#            print ("Don't know how to make a deep copy of (%d,%d), making shallow copy"%(i,j))
            print ("Don't know how to make a deep copy, making shallow copy")
            return copy.copy(obj)

def block_tensor(obj):
    """Return either a block_vec or a block_mat, depending on the shape of the object"""
    from . import block_mat, block_vec
    import numpy
    if isinstance(obj, (block_mat, block_vec)):
        return obj

    from ufl_legacy import Form
    if isinstance(obj, Form):
        from .splitting import split_form
        obj = split_form(obj)
    blocks = numpy.array(obj)
    if len(blocks.shape) == 2:
        return block_mat(blocks)
    elif len(blocks.shape) == 1:
        return block_vec(blocks)
    else:
        raise RuntimeError("Not able to create block container of rank %d"%len(blocks.shape))

def _create_vec(template, dim):
    from dolfin import DirichletBC, FunctionSpace, Function
    if dim is not None and hasattr(template, 'create_vec'):
        return template.create_vec(dim)
    if isinstance(template, DirichletBC):
        V = FunctionSpace(template.function_space())
    elif isinstance(template, FunctionSpace):
        V = template
    else:
        return None
    if V.component():
        return None
    return Function(V).vector()

def create_vec_from(templates, dim=None):
    """Try to create a dolfin vector from a (list of) templates.  A template is
    anything that we can retrieve a function space from (currently a
    FunctionSpace or a DirichletBC), or anything with a create_vec method (if
    dim is set).
    """
    for template in wrap_in_list(templates):
        v = _create_vec(template, dim)
        if v:
            return v
    raise ValueError("Unable to create vector from template")

def wrap_in_list(obj, types=object):
    """Make the argument into a list, suitable for iterating over. If it is
    already iterable, return it; if it is None, return the empty list; if it is
    a not iterable, return a length-one list. Optionally check the type."""
    if obj is None:
        lst = []
    elif hasattr(obj, '__iter__'):
        lst = list(obj)
    else:
        lst = [obj]
    for obj in lst:
        if not isinstance(obj, types):
            raise TypeError("expected a (list of) %s, not %s" % (types, type(obj)))
    return lst

def flatten(l):
    if isinstance(l, (list, tuple)):
        for el in l:
            for sub in flatten(el):
                yield sub
    else:
        yield l

def create_diagonal_matrix(V, val=1.0):
    from dolfin import TrialFunction, TestFunction
    from dolfin import assemble, Constant, inner, dx
    import numpy

    u,v = TrialFunction(V),TestFunction(V)
    Z = assemble(Constant(0)*inner(u,v)*dx)
    if val != 0.0:
        idx = numpy.arange(*Z.local_range(0), dtype=numpy.intc)
        Z.ident(idx)
        if val != 1.0:
            Z *= val
    return Z
