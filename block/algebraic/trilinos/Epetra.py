from __future__ import division

"""Module implementing algebraic operations on Epetra matrices: Diag, InvDiag
etc, as well as the collapse() method which performs matrix addition and
multiplication.
"""

from block.block_base import block_base

class diag_op(block_base):
    """Base class for diagonal Epetra operators (represented by an Epetra vector)."""
    from block.object_pool import shared_vec_pool

    def __init__(self, v):
        from PyTrilinos import Epetra
        assert isinstance(v, (Epetra.MultiVector, Epetra.Vector))
        self.v = v

    def transpose(self):
        return self

    def matvec(self, b):
        try:
            b_vec = b.down_cast().vec()
        except AttributeError:
            return NotImplemented

        x = self.create_vec()
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for %s matvec, %d != %d'%(self.__class__.__name__,len(x),len(b)))

        x.down_cast().vec().Multiply(1.0, self.v, b_vec, 0.0)
        return x

    transpmult = matvec

    def matmat(self, other):
        from PyTrilinos import Epetra
        try:
            from numpy import isscalar
            if isscalar(other):
                x = Epetra.Vector(self.v)
                x.Scale(other)
                return diag_op(other)
            other = other.down_cast()
            if hasattr(other, 'mat'):
                C = Epetra.CrsMatrix(other.mat())
                C.LeftScale(self.v)
                return matrix_op(C)
            else:
                x = Epetra.Vector(self.v.Map())
                x.Multiply(1.0, self.v, other.vec(), 0.0)
                return diag_op(x)
        except AttributeError:
            raise TypeError("can't extract matrix data from type '%s'"%str(type(other)))

    def add(self, other, lscale=1.0, rscale=1.0):
        from numpy import isscalar
        from PyTrilinos import Epetra
        try:
            if isscalar(other):
                x = Epetra.Vector(self.v.Map())
                x.PutScalar(other)
                other = diag_op(x)
            other = other.down_cast()
            if isinstance(other, matrix_op):
                return other.add(self)
            else:
                x = Epetra.Vector(self.v)
                x.Update(rscale, other.vec(), lscale)
                return diag_op(x)
        except AttributeError:
            raise TypeError("can't extract matrix data from type '%s'"%str(type(other)))

    @shared_vec_pool
    def create_vec(self, dim=1):
        from dolfin import EpetraVector
        if dim > 1:
            raise ValueError('dim must be <= 1')
        return EpetraVector(self.v.Map())

    def down_cast(self):
        return self
    def vec(self):
        return self.v

    def __str__(self):
        return '<%s %dx%d>'%(self.__class__.__name__,len(self.v),len(self.v))

class matrix_op(block_base):
    """Base class for Epetra operators (represented by an Epetra matrix)."""
    from block.object_pool import vec_pool

    def __init__(self, M, transposed=False):
        from PyTrilinos import Epetra
        assert isinstance(M, (Epetra.CrsMatrix, Epetra.FECrsMatrix))
        self.M = M
        self.transposed = transposed

    def transpose(self):
        return matrix_op(self.M, not self.transposed)

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        if self.transposed:
            domainlen = self.M.NumGlobalRows()
            x = self.create_vec(dim=1)
        else:
            domainlen = self.M.NumGlobalCols()
            x = self.create_vec(dim=0)
        if len(b) != domainlen:
            raise RuntimeError(
                'incompatible dimensions for %s matvec, %d != %d'%(self.__class__.__name__,domainlen,len(b)))
        self.M.SetUseTranspose(self.transposed)
        self.M.Apply(b.down_cast().vec(), x.down_cast().vec())
        self.M.SetUseTranspose(False) # May not be necessary?
        return x

    def transpmult(self, b):
        self.transposed = not self.transposed
        result = self.matvec(b)
        self.transposed = not self.transposed
        return result

    def matmat(self, other):
        from PyTrilinos import Epetra
        from numpy import isscalar
        try:
            if isscalar(other):
                C = Epetra.CrsMatrix(self.M)
                if other != 1:
                    C.Scale(other)
                return matrix_op(C, self.transposed)
            other = other.down_cast()
            if hasattr(other, 'mat'):
                from PyTrilinos import EpetraExt
                # Note: Tried ColMap for the transposed matrix, but that
                # crashes when the result is used by ML in parallel
                RowMap = self.M.DomainMap() if self.transposed else self.M.RowMap()
                C = Epetra.CrsMatrix(Epetra.Copy, RowMap, 100)
                assert (0 == EpetraExt.Multiply(self.M,      self.transposed,
                                                other.mat(), other.transposed,
                                                C))
                C.OptimizeStorage()
                return matrix_op(C)
            else:
                C = Epetra.CrsMatrix(self.M)
                C.RightScale(other.vec())
                return matrix_op(C, self.transposed)
        except AttributeError:
            raise TypeError("can't extract matrix data from type '%s'"%str(type(other)))

    def add(self, other, lscale=1.0, rscale=1.0):
        from PyTrilinos import Epetra
        try:
            other = other.down_cast()
            if hasattr(other, 'mat'):
                from PyTrilinos import EpetraExt
                C = Epetra.CrsMatrix(Epetra.Copy, self.M.RowMap(), 100)
                assert (0 == EpetraExt.Add(self.M,      self.transposed,      lscale, C, 0.0))
                assert (0 == EpetraExt.Add(other.mat(), other.transposed, rscale, C, 1.0))
                C.FillComplete()
                C.OptimizeStorage()
                return matrix_op(C)
            else:
                lhs = self.matmat(lscale)
                D = Diag(lhs).add(other, rscale=rscale)
                lhs.M.ReplaceDiagonalValues(D.vec())
                return lhs
        except AttributeError:
            raise TypeError("can't extract matrix data from type '%s'"%str(type(other)))

    @vec_pool
    def create_vec(self, dim=1):
        from dolfin import EpetraVector
        if dim == 0:
            m = self.M.RangeMap()
        elif dim == 1:
            m = self.M.DomainMap()
        else:
            raise ValueError('dim must be <= 1')
        return EpetraVector(m)

    def down_cast(self):
        return self
    def mat(self):
        return self.M

    def __str__(self):
        format = '<%s transpose(%dx%d)>' if self.transposed else '<%s %dx%d>'
        return format%(self.__class__.__name__, self.M.NumGlobalRows(), self.M.NumGlobalCols())

class Diag(diag_op):
    """Extract the diagonal entries of a matrix"""
    def __init__(self, A):
        from PyTrilinos import Epetra
        A = A.down_cast().mat()
        v = Epetra.Vector(A.RowMap())
        A.ExtractDiagonalCopy(v)
        diag_op.__init__(self, v)

class InvDiag(Diag):
    """Extract the inverse of the diagonal entries of a matrix"""
    def __init__(self, A):
        Diag.__init__(self, A)
        self.v.Reciprocal(self.v)

class LumpedInvDiag(diag_op):
    """Extract the inverse of the lumped diagonal of a matrix (i.e., sum of the
    absolute values in the row)"""
    def __init__(self, A):
        from PyTrilinos import Epetra
        A = A.down_cast().mat()
        v = Epetra.Vector(A.RowMap())
        A.InvRowSums(v)
        diag_op.__init__(self, v)

def _collapse(x):
    # Works by calling the matmat(), transpose() and add() methods of
    # diag_op/matrix_op, depending on the input types. The input is a tree
    # structure of block.block_mul objects, which is collapsed recursively.

    # This method knows too much about the internal variables of the
    # block_mul objects... should convert to accessor functions.

    from block.block_compose import block_mul, block_add, block_sub, block_transpose
    from block.block_mat import block_mat
    from numpy import isscalar
    from dolfin import GenericMatrix
    if isinstance(x, (matrix_op, diag_op)):
        return x
    elif isinstance(x, GenericMatrix):
        return matrix_op(x.down_cast().mat())
    elif isinstance(x, block_mat):
        if x.blocks.shape != (1,1):
            raise NotImplementedError("collapse() for block_mat with shape != (1,1)")
        return _collapse(x[0,0])
    elif isinstance(x, block_mul):
        factors = map(_collapse, reversed(x))
        while len(factors) > 1:
            A = factors.pop()
            B = factors.pop()
            if isscalar(A) and isscalar(B):
                C = A*B
            else:
                C = B.matmat(A) if isscalar(A) else A.matmat(B)
            factors.append(C)
        return factors[0]
    elif isinstance(x, block_add):
        A,B = map(_collapse, x)
        if isscalar(A) and isscalar(B):
            return A+B
        else:
            return B.add(A) if isscalar(A) else A.add(B)
    elif isinstance(x, block_sub):
        A,B = map(_collapse, x)
        if isscalar(A) and isscalar(B):
            return A-B
        else:
            return B.add(A, lscale=-1.0) if isscalar(A) else A.add(B, rscale=-1.0)
    elif isinstance(x, block_transpose):
        A = map(_collapse, x)
        return A if isscalar(A) else A.transpose()
    elif isscalar(x):
        return x
    else:
        raise NotImplementedError("collapse() for type '%s'"%str(type(x)))

def collapse(x):
    """Compute an explicit matrix representation of an operator. For example,
    given a block_ mul object M=A*B, collapse(M) performs the actual matrix
    multiplication.
    """
    # Since _collapse works recursively, this method is a user-visible wrapper
    # to print timing, and to check input/output arguments.
    from time import time
    from dolfin import info, warning
    T = time()
    res = _collapse(x)
    if getattr(res, 'transposed', False):
        # transposed matrices will normally be converted to a non-transposed
        # one by matrix multiplication or addition, but if the transpose is the
        # outermost operation then this doesn't work.
        warning('Transposed matrix returned from collapse() -- this matrix can be used for multiplications, '
                + 'but not (for example) as input to ML. Try to convert from (A*B)^T to B^T*A^T in your call.')
    info('computed explicit matrix representation %s in %.2f s'%(str(res),time()-T))
    return res


def create_identity(vec, val=1):
    """Create an identity matrix with the layout given by the supplied
    GenericVector. The values of the vector are NOT used."""
    import numpy
    from PyTrilinos import Epetra
    from dolfin import EpetraMatrix
    rowmap = vec.down_cast().vec().Map()
    graph = Epetra.CrsGraph(Epetra.Copy, rowmap, 1)
    indices = numpy.array([0], dtype=numpy.intc)
    for row in rowmap.MyGlobalElements():
        indices[0] = row
        graph.InsertGlobalIndices(row, indices)
    graph.FillComplete()

    matrix = EpetraMatrix(graph)
    indices = numpy.array(rowmap.MyGlobalElements(), dtype=numpy.uintc)
    if val == 0:
        matrix.zero(indices)
    else:
        matrix.ident(indices)
        if val != 1:
            matrix *= val

    return matrix
