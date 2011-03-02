from __future__ import division

from dolfin import Vector
from block.blockbase import blockbase

class ML(blockbase):
    def __init__(self, A, pdes=1):
        from PyTrilinos.ML import MultiLevelPreconditioner
        # create the ML preconditioner
        MLList = {
            #"max levels"                : 30,
#            "ML output"                 : 10,
            "smoother: type"            : "ML symmetric Gauss-Seidel" ,
            #"smoother: sweeps"          : 2,
            #"cycle applications"        : 2,
            #"prec type"                 : "MGW",
            "aggregation: type"         : "Uncoupled" ,
            #"PDE equations"             : pdes,
            "ML validate parameter list": True,
            }
        self.A = A # Prevent matrix being deleted
        self.ml_prec = MultiLevelPreconditioner(A.down_cast().mat(), 0)
        self.ml_prec.SetParameterList(MLList)
        self.ml_agg = self.ml_prec.GetML_Aggregate()
        err = self.ml_prec.ComputePreconditioner()
        if err:
            raise RuntimeError('ComputePreconditioner returned %d'%err)

    def matvec(self, b):
        if not isinstance(b, Vector):
            return NotImplemented
        # apply the ML preconditioner
        x = Vector(len(b))
        err = self.ml_prec.ApplyInverse(b.down_cast().vec(), x.down_cast().vec())
        if err:
            raise RuntimeError('ApplyInverse returned %d'%err)
        return x

    def down_cast(self):
        return self.ml_prec
