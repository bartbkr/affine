"""
Affine unit tests

For the following in the docs:
  L = number of lags in VAR process governing pricing kernel
  O = number of observed factors in VAR process governing pricing kernel
  U = number of unobserved, latent factors in VAR process governing
  pricing kernel
"""
from unittest import TestCase

import unittest
import copy
import numpy as np
import numpy.ma as ma
import pandas as pa

from affine.constructors.helper import make_nomask
from affine.model.affine import Affine
from affine.model.util import transform_var1

# parameters for running tests
test_size = 100
lags = 4
neqs = 5
nyields = 5
latent = 1

class TestInitialize(TestCase):
    """
    Tests for methods related to instantiation of a new Affine object
    """
    def setUp(self):

        np.random.seed(100)

        # initialize yield curve and VAR observed factors
        yc_data_test = pa.DataFrame(np.random.random((test_size - lags,
                                                      nyields)))
        var_data_test = pa.DataFrame(np.random.random((test_size, neqs)))
        mats = list(range(1, nyields + 1))

        # initialize masked arrays
        self.dim = dim = lags * neqs
        lam_0 = make_nomask([dim, 1])
        lam_1 = make_nomask([dim, dim])
        delta_0 = make_nomask([1, 1])
        delta_1 = make_nomask([dim, 1])
        mu = make_nomask([dim, 1])
        phi = make_nomask([dim, dim])
        sigma = make_nomask([dim, dim])

        # Setup some of the elements as non-zero
        # This sets up a fake model where only lambda_0 and lambda_1 are
        # estimated
        lam_0[:neqs] = ma.masked
        lam_1[:neqs, :neqs] = ma.masked
        delta_0[:, :] = np.random.random(1)
        delta_1[:neqs] = np.random.random((neqs, 1))
        mu[:neqs] = np.random.random((neqs, 1))
        phi[:neqs, :] = np.random.random((neqs, dim))
        sigma[:, :] = np.identity(dim)

        self.mod_kwargs = {
            'yc_data': yc_data_test,
            'var_data': var_data_test,
            'lags': lags,
            'neqs': neqs,
            'mats': mats,
            'lam_0_e': lam_0,
            'lam_1_e': lam_1,
            'delta_0_e': delta_0,
            'delta_1_e': delta_1,
            'mu_e': mu,
            'phi_e': phi,
            'sigma_e': sigma
        }

    def test_create_correct(self):
        """
        Tests whether __init__ successfully initializes an Affine model object.
        If the Affine object does not successfully instantiate, then this test
        fails, otherwise it passes.
        """
        model = Affine(**self.mod_kwargs)
        self.assertIsInstance(model, Affine)

    def test_wrong_lam0_size(self):
        """
        Tests whether size check asserts for lam_0_e is implemented
        correctly. If the lam_0_e parameter is not of the correct size,
        which is (L * O + U) by 1, then an assertion error should be raised,
        resulting in a passed test. If lam_0_e is of the incorrect size and
        no assertion error is raised, this test fails.
        """
        mod_kwargs = self.mod_kwargs
        # lam_0_e of incorrect size
        mod_kwargs['lam_0_e'] = make_nomask([self.dim - 1, 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_lam1_size(self):
        """
        Tests whether size check asserts for lam_1_e is implemented correctly.
        If the lam_1_e parameter is not of the correct size, which is (L
        * O + U) by (L * O + U), then an assertion error should be raised,
        resulting in a passed test.  If lam_1_e is of the incorrect size and no
        assertion error is raised, this test fails.
        """
        mod_kwargs = self.mod_kwargs
        # lam_1_e of incorrect size
        mod_kwargs['lam_1_e'] = make_nomask([self.dim - 1, self.dim + 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_delta_1_size(self):
        """
        Tests whether size check asserts for delta_1_e is implemented
        correctly. If the delta_1_e parameter is not of the correct size, which
        is (L * O + U) by 1, then an assertion error should be raised,
        resulting in a passed test. If delta_1_e is of the incorrect size and
        no assertion error is raised, this test fails.
        """
        mod_kwargs = self.mod_kwargs
        # delta_1_e of incorrect size
        mod_kwargs['delta_1_e'] = make_nomask([self.dim + 1, 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_mu_e_size(self):
        """
        Tests whether size check asserts for mu_e is implemented correctly.  If
        the mu_e parameter is not of the correct size, which is (L * O + U) by
        1, then an assertion error should be raised, resulting in a passed
        test. If mu_e is of the incorrect size and no assertion error is
        raised, this test fails.
        """
        mod_kwargs = self.mod_kwargs
        # mu_e of incorrect size
        mod_kwargs['mu_e'] = make_nomask([self.dim + 2, 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_phi_e_size(self):
        """
        Tests whether size check asserts for phi_e is implemented correctly.
        If the phi_e parameter is not of the correct size, which is (L * O + U)
        by (L * O + U), then an assertion error should be raised, resulting in
        a passed test. If phi_e is of the incorrect size and no assertion error
        is raised, this test fails.
        """
        mod_kwargs = self.mod_kwargs
        # phi_e of incorrect size
        mod_kwargs['phi_e'] = make_nomask([self.dim + 2, self.dim - 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_sigma_e_size(self):
        """
        Tests whether size check asserts for sigma_e is implemented correctly.
        If the sigma_e parameter is not of the correct size, which is (L
        * O + U) by (L * O + U), then an assertion error should be raised,
        resulting in a passed test. If sigma_e is of the incorrect size and no
        assertion error is raised, this test fails.
        """
        mod_kwargs = self.mod_kwargs
        # sigma_e of incorrect size
        mod_kwargs['sigma_e'] = make_nomask([self.dim - 2, self.dim])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_var_data_nulls(self):
        """
        Tests if nulls appear in var_data whether an AssertionError is raised.
        If any nulls appear in var_data and an AssertionError is raised, the
        test passes. Otherwise if nulls are passed in and an AssertionError is
        not raised, the test fails.
        """
        mod_kwargs = self.mod_kwargs
        # replace a value in var_data with null
        mod_kwargs['var_data'][1, 1] = np.nan
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_yc_data_nulls(self):
        """
        Tests if nulls appear in yc_data whether AssertionError is raised.  If
        any nulls appear in yc_data and an AssertionError is raised, the test
        passes. Otherwise if nulls are passed in and an AssertionError is not
        raised, the test fails.
        """
        mod_kwargs = self.mod_kwargs
        # replace a value in var_data with null
        mod_kwargs['yc_data'][1, 1] = np.nan
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_no_estimated_values(self):
        """
        Tests if AssertionError is raised if there are no masked values in
        the estimation arrays, implying no parameters to be estimated. If
        the object passed in has no estimated values and an AssertionError
        is raised, the test passes. Otherwise if no estimated values are
        passed in and an AssertionError is not raised, the test fails.
        """
        mod_kwargs = self.mod_kwargs
        # replace a value in var_data with null
        mod_kwargs['lam_0_e'] = make_nomask([self.dim, 1])
        mod_kwargs['lam_1_e'] = make_nomask([self.dim, self.dim])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_input_arrays_inconsistently_typed(self):
        """
        Tests if check is performed whether the  input arrays are all of the
        same type.
        """
        mod_kwargs = copy.copy(self.mod_kwargs)
        # DOC: Make each of them complex
        mod_kwargs['lam_1_e'] = mod_kwargs['lam_1_e'] + 1j
        self.assertRaises(AssertionError, Affine, **mod_kwargs)
        mod_kwargs = copy.copy(self.mod_kwargs)
        mod_kwargs['delta_0_e'] = mod_kwargs['delta_0_e'] + 1j
        self.assertRaises(AssertionError, Affine, **mod_kwargs)
        mod_kwargs = copy.copy(self.mod_kwargs)
        mod_kwargs['delta_1_e'] = mod_kwargs['delta_1_e'] + 1j
        self.assertRaises(AssertionError, Affine, **mod_kwargs)
        mod_kwargs = copy.copy(self.mod_kwargs)
        mod_kwargs['mu_e'] = mod_kwargs['mu_e'] + 1j
        self.assertRaises(AssertionError, Affine, **mod_kwargs)
        mod_kwargs = copy.copy(self.mod_kwargs)
        mod_kwargs['phi_e'] = mod_kwargs['phi_e'] + 1j
        self.assertRaises(AssertionError, Affine, **mod_kwargs)
        mod_kwargs = copy.copy(self.mod_kwargs)
        mod_kwargs['sigma_e'] = mod_kwargs['sigma_e'] + 1j
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

class TestEstimationSupportMethods(TestCase):
    """
    Tests for support methods related to estimating models
    """
    def setUp(self):

        np.random.seed(100)

        # initialize yield curve and VAR observed factors
        yc_data_test = pa.DataFrame(np.random.random((test_size - lags,
                                                      nyields)))
        var_data_test = pa.DataFrame(np.random.random((test_size, neqs)))
        mats = list(range(1, nyields + 1))

        # initialize masked arrays
        self.dim = dim = lags * neqs + latent
        lam_0 = make_nomask([dim, 1])
        lam_1 = make_nomask([dim, dim])
        delta_0 = make_nomask([1, 1])
        delta_1 = make_nomask([dim, 1])
        mu = make_nomask([dim, 1])
        phi = make_nomask([dim, dim])
        sigma = make_nomask([dim, dim])

        # Setup some of the elements as non-zero
        # This sets up a fake model where only lambda_0 and lambda_1 are
        # estimated
        lam_0[:neqs] = ma.masked
        lam_0[-latent:] = ma.masked
        lam_1[:neqs, :neqs] = ma.masked
        lam_1[-latent:, -latent:] = ma.masked
        delta_0[:, :] = np.random.random(1)
        delta_1[:neqs] = np.random.random((neqs, 1))
        mu[:neqs] = np.random.random((neqs, 1))
        phi[:neqs, :] = np.random.random((neqs, dim))
        sigma[:, :] = np.identity(dim)

        self.mod_kwargs = {
            'yc_data': yc_data_test,
            'var_data': var_data_test,
            'lags': lags,
            'neqs': neqs,
            'mats': mats,
            'lam_0_e': lam_0,
            'lam_1_e': lam_1,
            'delta_0_e': delta_0,
            'delta_1_e': delta_1,
            'mu_e': mu,
            'phi_e': phi,
            'sigma_e': sigma,
            'latent': latent,
            'no_err': [1]
        }

        self.guess_params = np.random.random((neqs**2 + neqs + (2 * latent),)
                                            ).tolist()
        self.affine_obj = Affine(**self.mod_kwargs)

    def test_loglike(self):
        """
        Tests if loglikelihood is calculated. If the loglikelihood is
        calculated given a set of parameters, then this test passes.
        Otherwise, it fails.
        """
        self.affine_obj.loglike(self.guess_params)

    def test_score(self):
        """
        Tests if score of the likelihood is calculated. If the score
        calculation succeeds without error, then the test passes. Otherwise,
        the test fails.
        """
        self.affine_obj.score(self.guess_params)

    def test_hessian(self):
        """
        Tests if hessian of the likelihood is calculated. If the hessian
        calculation succeeds without error, then the test passes. Otherwise,
        the test fails.
        """
        self.affine_obj.hessian(self.guess_params)

    def test_std_errs(self):
        """
        Tests if standard errors are calculated. If the standard error
        calculation succeeds, then the test passes. Otherwise, the test
        fails.
        """
        self.affine_obj.std_errs(self.guess_params)

    def test_params_to_array(self):
        """
        Tests if the params_to_array function works correctly, with and without
        returning masked arrays. In order to pass, the params_to_array function
        must return masked arrays with the masked elements filled in when the
        return_mask argument is set to True and contiguous standard numpy
        arrays when the return_mask argument is False. Otherwise, the test
        fails.
        """
        arrays_no_mask = self.affine_obj.params_to_array(self.guess_params)
        for arr in arrays_no_mask[:-1]:
            self.assertIsInstance(arr, np.ndarray)
            self.assertNotIsInstance(arr, np.ma.core.MaskedArray)
        arrays_w_mask = self.affine_obj.params_to_array(self.guess_params,
                                                        return_mask=True)
        for arr in arrays_w_mask[:-1]:
            self.assertIsInstance(arr, np.ma.core.MaskedArray)

    def test_params_to_array_inconsistent_types(self):
        """
        Tests if an assertion error is raised when parameters of different
        types are passed in
        """
        guess_params_adj = self.guess_params
        guess_params_adj[-1] = np.complex_(guess_params_adj[-1])
        self.assertRaises(AssertionError, self.affine_obj.params_to_array,
                          guess_params_adj)

    def test_params_to_array_zeromask(self):
        """
        Tests if params_to_array_zeromask function works correctly. In order to
        pass, params_to_array_zeromask must return masked arrays with the
        guess_params elements that are zero unmasked and set to zero in the
        appropriate arrays. The new guess_params array is also returned with
        those that were 0 removed. If both of these are not returned correctly,
        the test fails.
        """
        guess_params_arr = np.array(self.guess_params)
        neqs = self.affine_obj.neqs
        guess_params_arr[:neqs] = 0
        guess_params = guess_params_arr.tolist()
        guess_length = self.affine_obj._gen_guess_length()
        params_guesses = self.affine_obj.params_to_array_zeromask(guess_params)
        updated_guesses = params_guesses[-1]
        self.assertEqual(len(updated_guesses), len(guess_params) - neqs)

        # ensure that number of masked has correctly been set
        count_masked_new = ma.count_masked(params_guesses[0])
        count_masked_orig = ma.count_masked(self.affine_obj.lam_0_e)
        self.assertEqual(count_masked_new, count_masked_orig - neqs)

    def test_gen_pred_coef(self):
        """
        Tests if Python-driven gen_pred_coef function runs. If a set of
        parameter arrays are passed into the gen_pred_coef function and the
        A and B arrays are returned, then the test passes. Otherwise, the test
        fails.
        """
        params = self.affine_obj.params_to_array(self.guess_params)
        self.affine_obj.gen_pred_coef(*params[:-1])

    def test_opt_gen_pred_coef(self):
        """
        Tests if C-driven gen_pred_coef function runs. If a set of parameter
        arrays are passed into the opt_gen_pred_coef function and the A and
        B arrays are return, then the test passes. Otherwise, the test fails.
        """
        params = self.affine_obj.params_to_array(self.guess_params)
        self.affine_obj.opt_gen_pred_coef(*params)

    def test_py_C_gen_pred_coef_equal(self):
        """
        Tests if the Python-driven and C-driven gen_pred_coef functions produce
        the same result, up to a precision of 1e-14. If the gen_pred_coef and
        opt_gen_pred_coef functions produce the same result, then the test
        passes. Otherwise, the test fails.
        """
        params = self.affine_obj.params_to_array(self.guess_params)
        py_gpc = self.affine_obj.gen_pred_coef(*params[:-1])
        c_gpc = self.affine_obj.opt_gen_pred_coef(*params)
        for aix, array in enumerate(py_gpc):
            np.testing.assert_allclose(array, c_gpc[aix], rtol=1e-14)

    def test__solve_unobs(self):
        """
        Tests if the _solve_unobs function runs. If the _solve_unobs function
        runs and the latent series, likelihood jacobian, and yield errors are
        returned, then the test passes. Otherwise the test fails.
        """
        guess_params = self.guess_params
        param_arrays = self.affine_obj.params_to_array(guess_params)
        a_in, b_in = self.affine_obj.gen_pred_coef(*param_arrays[:-1])
        result = self.affine_obj._solve_unobs(a_in=a_in, b_in=b_in)

    def test__affine_pred(self):
        """
        Tests if the _affine_pred function runs. If the affine_pred function
        produces a list of the yields stacked in order of increasing maturity
        and is of the expected shape, the test passes. Otherwise, the test
        fails.
        """
        lat = self.affine_obj.lat
        yobs = self.affine_obj.yobs
        mats = self.affine_obj.mats
        var_data_vert_tpose = self.affine_obj.var_data_vert.T

        guess_params = self.guess_params
        latent_rows = np.random.random((lat, yobs))
        data = np.append(var_data_vert_tpose, latent_rows, axis=0)
        pred = self.affine_obj._affine_pred(data, *guess_params)
        self.assertEqual(len(pred), len(mats) * yobs)

    def test__gen_mat_list(self):
        """
        Tests if _gen_mat_list generates a length 2 tuple with a list of the
        maturities estimated without error followed by those estimated with
        error. If _gen_mat_list produces a tuple of lists of those yields
        estimates without error and then those with error, this test passes.
        Otherwise, the test fails.
        """
        no_err_mat, err_mat = self.affine_obj._gen_mat_list()
        self.assertEqual(no_err_mat, [2])
        self.assertEqual(err_mat, [1,3,4,5])

class TestEstimationSupportMethodsComplex(TestCase):
    """
    Test cases where instantiation is complex
    """
    def setUp(self):

        np.random.seed(100)

        # initialize yield curve and VAR observed factors
        yc_data_test = pa.DataFrame(np.random.random((test_size - lags,
                                                      nyields)))
        var_data_test = pa.DataFrame(np.random.random((test_size, neqs)))
        mats = list(range(1, nyields + 1))

        # initialize masked arrays
        self.dim = dim = lags * neqs + latent
        lam_0 = make_nomask([dim, 1]) + 0j
        lam_1 = make_nomask([dim, dim]) + 0j
        delta_0 = make_nomask([1, 1]) + 0j
        delta_1 = make_nomask([dim, 1]) + 0j
        mu = make_nomask([dim, 1]) + 0j
        phi = make_nomask([dim, dim]) + 0j
        sigma = make_nomask([dim, dim]) + 0j

        # Setup some of the elements as non-zero
        # This sets up a fake model where only lambda_0 and lambda_1 are
        # estimated
        lam_0[:neqs] = ma.masked
        lam_0[-latent:] = ma.masked
        lam_1[:neqs, :neqs] = ma.masked
        lam_1[-latent:, -latent:] = ma.masked
        delta_0[:, :] = np.random.random(1)
        delta_1[:neqs] = np.random.random((neqs, 1))
        mu[:neqs] = np.random.random((neqs, 1))
        phi[:neqs, :] = np.random.random((neqs, dim))
        sigma[:, :] = np.identity(dim)

        self.mod_kwargs = {
            'yc_data': yc_data_test,
            'var_data': var_data_test,
            'lags': lags,
            'neqs': neqs,
            'mats': mats,
            'lam_0_e': lam_0,
            'lam_1_e': lam_1,
            'delta_0_e': delta_0,
            'delta_1_e': delta_1,
            'mu_e': mu,
            'phi_e': phi,
            'sigma_e': sigma,
            'latent': latent,
            'no_err': [1]
        }

        self.guess_params = np.random.random((neqs**2 + neqs + (2 * latent),)
                                            ).tolist()
        self.affine_obj = Affine(**self.mod_kwargs)

    def test_opt_gen_pred_coef_float(self):
        """
        Tests if complex values are return properly
        """
        # DOC: Need to make sure that the arrays are also np.complex
        guess_params = self.guess_params
        params = self.affine_obj.params_to_array(guess_params)
        arrays_gpc = self.affine_obj.opt_gen_pred_coef(*params)
        for array in arrays_gpc:
            self.assertEqual(array.dtype, np.complex_)

    def test_opt_gen_pred_coef_complex(self):
        """
        Tests if complex values are return properly
        """
        # DOC: Need to make sure that the arrays are also np.complex
        guess_params = [np.complex_(el) for el in self.guess_params]
        params = self.affine_obj.params_to_array(guess_params)
        arrays_gpc = self.affine_obj.opt_gen_pred_coef(*params)
        for array in arrays_gpc:
            self.assertEqual(array.dtype, np.complex_)

class TestEstimationMethods(TestCase):
    """
    Tests for solution methods
    """
    def setUp(self):

        ## Non-linear least squares
        np.random.seed(101)

        # initialize yield curve and VAR observed factors
        yc_data_test = pa.DataFrame(np.random.random((test_size - lags,
                                                      nyields)))
        var_data_test = pa.DataFrame(np.random.random((test_size, neqs)))
        mats = list(range(1, nyields + 1))

        # initialize masked arrays
        self.dim_nolat = dim = lags * neqs
        lam_0 = make_nomask([dim, 1])
        lam_1 = make_nomask([dim, dim])
        delta_0 = make_nomask([1, 1])
        delta_1 = make_nomask([dim, 1])
        mu = make_nomask([dim, 1])
        phi = make_nomask([dim, dim])
        sigma = make_nomask([dim, dim])

        # Setup some of the elements as non-zero
        # This sets up a fake model where only lambda_0 and lambda_1 are
        # estimated
        lam_0[:neqs] = ma.masked
        lam_1[:neqs, :neqs] = ma.masked
        delta_0[:, :] = np.random.random(1)
        delta_1[:neqs] = np.random.random((neqs, 1))
        mu[:neqs] = np.random.random((neqs, 1))
        phi[:neqs, :] = np.random.random((neqs, dim))
        sigma[:, :] = np.identity(dim)

        self.mod_kwargs_nolat = {
            'yc_data': yc_data_test,
            'var_data': var_data_test,
            'lags': lags,
            'neqs': neqs,
            'mats': mats,
            'lam_0_e': lam_0,
            'lam_1_e': lam_1,
            'delta_0_e': delta_0,
            'delta_1_e': delta_1,
            'mu_e': mu,
            'phi_e': phi,
            'sigma_e': sigma
        }

        self.guess_params_nolat = np.random.random((neqs**2 + neqs)).tolist()
        self.affine_obj_nolat = Affine(**self.mod_kwargs_nolat)

        ## Maximum likelihood build

        # initialize masked arrays
        self.dim_lat = dim = lags * neqs + latent
        lam_0 = make_nomask([dim, 1])
        lam_1 = make_nomask([dim, dim])
        delta_0 = make_nomask([1, 1])
        delta_1 = make_nomask([dim, 1])
        mu = make_nomask([dim, 1])
        phi = make_nomask([dim, dim])
        sigma = make_nomask([dim, dim])

        # Setup some of the elements as non-zero
        # This sets up a fake model where only lambda_0 and lambda_1 are
        # estimated
        lam_0[:neqs] = ma.masked
        lam_0[-latent:] = ma.masked
        lam_1[:neqs, :neqs] = ma.masked
        lam_1[-latent:, -latent:] = ma.masked
        delta_0[:, :] = np.random.random(1)
        delta_1[:neqs] = np.random.random((neqs, 1))
        mu[:neqs] = np.random.random((neqs, 1))
        phi[:neqs, :] = np.random.random((neqs, dim))
        sigma[:, :] = np.identity(dim)

        self.mod_kwargs = {
            'yc_data': yc_data_test,
            'var_data': var_data_test,
            'lags': lags,
            'neqs': neqs,
            'mats': mats,
            'lam_0_e': lam_0,
            'lam_1_e': lam_1,
            'delta_0_e': delta_0,
            'delta_1_e': delta_1,
            'mu_e': mu,
            'phi_e': phi,
            'sigma_e': sigma,
            'latent': latent,
            'no_err': [1]
        }

        self.guess_params_lat = np.random.random((neqs**2 + neqs +
                                                 (2 * latent),)).tolist()
        self.affine_obj_lat = Affine(**self.mod_kwargs)


    def test_solve_nls(self):
        """
        Tests whether or not basic estimation is performed for non-linear least
        squares case without any latent factors. If the numerical approximation
        method converges, this test passes. Otherwise, the test fails.
        """
        guess_params = self.guess_params_nolat
        method = 'nls'
        self.affine_obj_nolat.solve(guess_params, method=method, alg='newton',
                                    xtol=0.1, ftol=0.1)

    def test_solve_ml(self):
        """
        Tests whether or not model estimation converges is performed for direct
        maximum likelihood with a single latent factor. If the numerical
        approximation method converges, this test passes. Otherwise, the test
        fails.
        """
        guess_params = self.guess_params_lat
        method = 'ml'
        self.affine_obj_lat.solve(guess_params, method=method, alg='bfgs',
                                  xtol=0.1, ftol=0.1)

    ##Need test related to Kalman filter method

class TestResultsClass(TestCase):
    @classmethod
    def setUpClass(self):

        ## Non-linear least squares
        np.random.seed(100)

        # initialize yield curve and VAR observed factors
        yc_data_test = pa.DataFrame(np.random.random((test_size - lags,
                                                      nyields)))
        var_data_test = self.var_data_test = \
            pa.DataFrame(np.random.random((test_size, neqs)))
        self.mats = mats = list(range(1, nyields + 1))

        # initialize masked arrays
        self.dim_nolat = dim = lags * neqs
        lam_0 = make_nomask([dim, 1])
        lam_1 = make_nomask([dim, dim])
        delta_0 = make_nomask([1, 1])
        delta_1 = make_nomask([dim, 1])
        mu = make_nomask([dim, 1])
        phi = make_nomask([dim, dim])
        sigma = make_nomask([dim, dim])

        # Setup some of the elements as non-zero
        # This sets up a fake model where only lambda_0 and lambda_1 are
        # estimated
        lam_0[:neqs] = ma.masked
        lam_1[:neqs, :neqs] = ma.masked
        delta_0[:, :] = np.random.random(1)
        delta_1[:neqs] = np.random.random((neqs, 1))
        mu[:neqs] = np.random.random((neqs, 1))
        phi[:neqs, :] = np.random.random((neqs, dim))
        sigma[:, :] = np.identity(dim)

        self.mod_kwargs_nolat = {
            'yc_data': yc_data_test,
            'var_data': var_data_test,
            'lags': lags,
            'neqs': neqs,
            'mats': mats,
            'lam_0_e': lam_0,
            'lam_1_e': lam_1,
            'delta_0_e': delta_0,
            'delta_1_e': delta_1,
            'mu_e': mu,
            'phi_e': phi,
            'sigma_e': sigma
        }

        guess_params_nolat = np.random.random((neqs**2 + neqs)).tolist()
        affine_obj_nolat = Affine(**self.mod_kwargs_nolat)

        self.results = affine_obj_nolat.solve(guess_params_nolat, method='nls',
                                              xtol=0.1, ftol=0.1)

    def test_predicted_yields(self):
        """
        Tests whether the predicted yields are generated and are of the
        expected shape.
        """
        results = self.results
        pred = results.predicted_yields
        self.assertEqual(pred.shape, (test_size - lags, nyields))
        mats_check = [str(mat) + '_pred' for mat in self.mats]
        self.assertEqual(mats_check, pred.columns.tolist())

    def test_risk_neutral_yields(self):
        """
        Tests whether the risk-neurtral predicted yields are generated and are
        of the expected shape.
        """
        results = self.results
        rn = results.risk_neutral_yields
        self.assertEqual(rn.shape, (test_size - lags, nyields))
        mats_check = [str(mat) + '_risk_neutral' for mat in self.mats]
        self.assertEqual(mats_check, rn.columns.tolist())

    def test_term_premia(self):
        """
        Tests whether the term premia are generated and are of the expected
        shape.
        """
        results = self.results
        tp = results.term_premia
        self.assertEqual(tp.shape, (test_size - lags, nyields))
        mats_check = [str(mat) + '_tp' for mat in self.mats]
        self.assertEqual(mats_check, tp.columns.tolist())

    def test_generate_yields(self):
        """
        Tests whether the generated yields are generated given
        """
        results = self.results

        #standard case
        var_data_test = self.var_data_test[-15:]
        generate_yields = results.generate_yields(var_data_test,
                                                  adjusted=False)
        predicted_sset = results.predicted_yields[-15 + lags:]
        self.assertTrue(np.all(generate_yields.values == predicted_sset.values))
        cols_check = [str(mat) + '_pred' for mat in self.mats]
        self.assertEqual(cols_check, generate_yields.columns.tolist())

        # #adjusted case
        # var_data_test =
        var_data_trans = transform_var1(self.var_data_test[-15:], lags)
        generate_yields = results.generate_yields(var_data_trans,
                                                  adjusted=True)
        self.assertTrue(np.all(generate_yields.values == predicted_sset.values))
        cols_check = [str(mat) + '_pred' for mat in self.mats]
        self.assertEqual(cols_check, generate_yields.columns.tolist())

if __name__ == '__main__':
    unittest.main()
