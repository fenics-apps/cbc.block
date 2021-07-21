from block.block_base import block_base
from builtins import str
from dolfin import as_backend_type
import haznics


def PETSc_to_dCSRmat(A):
    """
    Change data type for matrix (PETSc or dolfin matrix to dCSRmat pointer)
    """
    petsc_mat = as_backend_type(A).mat()

    # NB! store copies for now
    csr0 = petsc_mat.getValuesCSR()[0]
    csr1 = petsc_mat.getValuesCSR()[1]
    csr2 = petsc_mat.getValuesCSR()[2]

    return haznics.create_matrix(csr2, csr1, csr0)


class Precond(block_base):
    """
    Class of general preconditioners from HAZmath using SWIG

    """

    def __init__(self, A, prectype, parameters={}, precond=None):
        # haznics.dCSRmat* type (assert?)
        self.A = A
        # python dictionary of parameters
        self.parameters = parameters

        # init and set preconditioner (precond *)
        if precond:
            self.precond = precond
        else:
            import warnings
            warnings.warn(
                "!! Preconditioner not specified !! Creating default UA-AMG "
                "precond... ",
                RuntimeWarning)
            # change data type for the matrix (to dCSRmat pointer)
            A_ptr = PETSc_to_dCSRmat(A)

            # initialize amg parameters (AMG_param pointer)
            amgparam = haznics.amg_param_alloc(1)

            # print (relevant) amg parameters
            haznics.param_amg_print(amgparam)

            self.precond = haznics.create_precond(A_ptr, amgparam)

            # if fail, setup returns null
            if not precond:
                raise RuntimeError(
                    "AMG levels failed to set up (null pointer returned) ")

        # preconditioner type (string)
        self.prectype = prectype

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented

        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError('incompatible dimensions for matvec, %d != %d'
                               % (len(x), len(b)))

        # convert rhs and dx to numpy arrays
        b_np = b[:]
        x_np = x[:]

        # apply the preconditioner (solution dx saved in x_np)
        haznics.apply_precond(b_np, x_np, self.precond)

        # convert dx to GenericVector
        x.set_local(x_np)

        return x

    # noinspection PyMethodMayBeStatic
    def down_cast(self):
        return NotImplemented

    def __str__(self):
        return '<%s prec of %s>' % (self.__class__.__name__, str(self.A))


class AMG(Precond):
    """
    AMG preconditioner from the HAZmath Library with SWIG

    """

    def __init__(self, A, parameters={}):
        # change data type for the matrix (to dCSRmat pointer)
        A_ptr = PETSc_to_dCSRmat(A)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.amg_param_alloc(1)

        # set extra amg parameters
        if parameters:
            for key in parameters:
                if isinstance(parameters[key], str):
                    exec("amgparam.%s = \"%s\"" % (key, parameters[key]))
                # elif isinstance(parameters[key], function):
                #     haznics.py_callback_setup(parameters[key], amgparam)
                else:
                    exec("amgparam.%s = %s" % (key, parameters[key]))

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # set AMG preconditioner
        precond = haznics.create_precond_amg(A_ptr, amgparam)

        # if fail, setup returns null
        if not precond:
            raise RuntimeError(
                "AMG levels failed to set up (null pointer returned) ")

        Precond.__init__(self, A, "amg", parameters, precond)


class FAMG(Precond):
    """
    AMG preconditioner from the HAZmath Library

    """

    def __init__(self, A, M, parameters={'fpwr': 0.5, 'smoother': 'fjacobi'}):
        # change data type for the matrices (to dCSRmat pointer)
        A_ptr = PETSc_to_dCSRmat(A)
        M_ptr = PETSc_to_dCSRmat(M)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.amg_param_alloc(1)

        # set extra amg parameters
        if parameters:
            for key in parameters:
                if isinstance(parameters[key], str):
                    exec("amgparam.%s = \"%s\"" % (key, parameters[key]))
                else:
                    exec("amgparam.%s = %s" % (key, parameters[key]))

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # set AMG preconditioner
        precond = haznics.create_precond_famg(A_ptr, M_ptr, amgparam)

        # if fail, setup returns null
        if not precond:
            raise RuntimeError(
                "FAMG levels failed to set up (null pointer returned) ")

        Precond.__init__(self, A, "famg", parameters, precond)


class RA(Precond):
    """
    Rational approximation preconditioner from the HAZmath library

    """

    def __init__(self, A, M, dim=2,
                 parameters={'coefs': [1.0, 0.0], 'pwrs': [0.5, 0.0]}):

        # change data type for the matrices (to dCSRmat pointer)
        A_ptr = PETSc_to_dCSRmat(A)
        M_ptr = PETSc_to_dCSRmat(M)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.amg_param_alloc(1)

        # set extra amg parameters
        if parameters:
            for key in parameters:
                if isinstance(parameters[key], str):
                    exec("amgparam.%s = \"%s\"" % (key, parameters[key]))
                else:
                    exec("amgparam.%s = %s" % (key, parameters[key]))

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # get scalings
        scaling_a = 1. / A.norm("linf")
        scaling_m = 1. / as_backend_type(M).mat().getDiagonal().min()[1]

        # get coefs and powers
        alpha, beta = parameters['coefs']
        s_power, t_power = parameters['pwrs']

        # set RA preconditioner #
        precond = haznics.create_precond_ra(A_ptr, M_ptr, s_power, t_power,
                                            alpha, beta, scaling_a, scaling_m,
                                            amgparam)

        # if fail, setup returns null
        if not precond:
            raise RuntimeError(
                "Rational Approximation data failed to set up (null pointer "
                "returned) ")

        Precond.__init__(self, A, "RA", parameters, precond)


class HXCurl(Precond):
    """
    HX preconditioner from the HAZmath library for the curl-curl inner product
    NB! only for 3D problems
    TODO: needs update and test
    """

    def __init__(self, Acurl, Pcurl, Grad, parameters={}):
        # change data type for the matrices (to dCSRmat pointer)
        Acurl_ptr = PETSc_to_dCSRmat(Acurl)
        Pcurl_ptr = PETSc_to_dCSRmat(Pcurl)
        Grad_ptr = PETSc_to_dCSRmat(Grad)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.amg_param_alloc(1)

        # set extra amg parameters
        if parameters:
            for key in parameters:
                if isinstance(parameters[key], str):
                    exec("amgparam.%s = \"%s\"" % (key, parameters[key]))
                else:
                    exec("amgparam.%s = %s" % (key, parameters[key]))

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # add or multi
        try:
            prectype = parameters['prectype']
        except KeyError:
            prectype = haznics.PREC_HX_CURL_A

        # set HX CURL preconditioner (NB: this sets up both data and fct)
        precond = haznics.create_precond_hxcurl(Acurl_ptr, Pcurl_ptr, Grad_ptr,
                                                prectype, amgparam)

        # if fail, setup returns null
        if not precond:
            raise RuntimeError(
                "HXcurl data failed to set up (null pointer returned) ")
        """
        try:
            prectype = parameters['prectype']
        except KeyError:
            prectype = ''

        if prectype in ["add", "Add", "ADD", "additive", "ADDITIVE"]:
            precond.fct = haznics.precond_hx_curl_additive
        elif prectype in ["multi", "MULTI", "Multi", "multiplicative",
                          "MULTIPLICATIVE"]:
            precond.fct = haznics.precond_hx_curl_multiplicative
        else:  # default is additive
            precond.fct = haznics.precond_hx_curl_additive
        """
        Precond.__init__(self, Acurl, "HXCurl_add", parameters, precond)


class HXDiv(Precond):
    """
    HX preconditioner from the HAZmath library for the div-div inner product
    TODO: needs update and test
    """

    def __init__(self, Adiv, Pdiv, Curl, Pcurl=None,
                 parameters={'dimension': 2}):
        # change data type for the matrices (to dCSRmat pointer)
        Adiv_ptr = PETSc_to_dCSRmat(Adiv)
        Pdiv_ptr = PETSc_to_dCSRmat(Pdiv)
        Curl_ptr = PETSc_to_dCSRmat(Curl)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.amg_param_alloc(1)

        # set extra amg parameters
        if parameters:
            for key in parameters:
                if isinstance(parameters[key], str):
                    exec("amgparam.%s = \"%s\"" % (key, parameters[key]))
                else:
                    exec("amgparam.%s = %s" % (key, parameters[key]))

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # get dimension and type of HX precond application
        try:
            dim = parameters['dimension']
        except KeyError:
            dim = 2

        # add or multi
        try:
            prectype = parameters['prectype']
        except KeyError:
            prectype = haznics.PREC_HX_DIV_A

        if dim == 3:
            # check Pcurl
            assert Pcurl, "For 3D case, Pcurl operator is needed!"

            # change data type for the Pcurl matrix (to dCSRmat pointer)
            Pcurl_ptr = PETSc_to_dCSRmat(Pcurl)

            # set HX DIV preconditioner (NB: this sets up both data and fct)
            precond = haznics.create_precond_hxdiv_3D(Adiv_ptr, Pdiv_ptr,
                                                      Curl_ptr, Pcurl_ptr,
                                                      prectype, amgparam)

            # if fail, setup returns null
            if not precond:
                raise RuntimeError(
                    "HXdiv data failed to set up (null pointer returned) ")
            """
            if prectype in ["add", "Add", "ADD", "additive", "ADDITIVE"]:
                precond.fct = haznics.precond_hx_div_additive

            elif prectype in ["multi", "MULTI", "Multi", "multiplicative", 
            "MULTIPLICATIVE"]:
                precond.fct = haznics.precond_hx_div_multiplicative

            else:
                # default is additive
                precond.fct = haznics.precond_hx_div_additive
            """
            Precond.__init__(self, Adiv, "HXDiv_add", parameters, precond)

        else:
            # set HX DIV preconditioner (NB: this sets up both data and fct)
            precond = haznics.create_precond_hxdiv_2D(Adiv_ptr, Pdiv_ptr,
                                                      Curl_ptr, prectype,
                                                      amgparam)

            # if fail, setup returns null
            if not precond:
                raise RuntimeError(
                    "HXdiv data failed to set up (null pointer returned) ")
            """
            if prectype in ["add", "Add", "ADD", "additive", "ADDITIVE"]:
                precond.fct = haznics.precond_hx_div_additive_2D

            elif prectype in ["multi", "MULTI", "Multi", "multiplicative", "MULTIPLICATIVE"]:
                precond.fct = haznics.precond_hx_div_multiplicative_2D

            else:
                # default is additive
                precond.fct = haznics.precond_hx_div_additive_2D
            """
            Precond.__init__(self, Adiv, "HXDiv_add", parameters, precond)

# ----------------------------------- EOF ----------------------------------- #