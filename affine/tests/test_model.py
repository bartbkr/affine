from unittest import TestCase

import unittest
import numpy as np
import numpy.ma as ma
import pandas as pa

from affine.constructors.helper import make_nomask
from affine.model.affine import Affine

# parameters for running tests
test_size = 100
lags = 4
neqs = 5
nyields = 5
latent = 1

class TestInitiatilize(TestCase):
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
        Tests whether __init__ successfully initializes model object.
        """
        model = Affine(**self.mod_kwargs)
        self.assertIsInstance(model, Affine)

    def test_wrong_lam0_size(self):
        """
        Tests whether size check asserts for lam_0_e is implemented correctly.
        """
        mod_kwargs = self.mod_kwargs
        # lam_0_e of incorrect size
        mod_kwargs['lam_0_e'] = make_nomask([self.dim - 1, 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_lam1_size(self):
        """
        Tests whether size check asserts for lam_1_e is implemented correctly.
        """
        mod_kwargs = self.mod_kwargs
        # lam_1_e of incorrect size
        mod_kwargs['lam_1_e'] = make_nomask([self.dim - 1, self.dim + 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_delta_1_size(self):
        """
        Tests whether size check asserts for delta_1_e is implemented
        correctly.
        """
        mod_kwargs = self.mod_kwargs
        # delta_1_e of incorrect size
        mod_kwargs['delta_1_e'] = make_nomask([self.dim + 1, 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_mu_e_size(self):
        """
        Tests whether size check asserts for mu_e is implemented correctly.
        """
        mod_kwargs = self.mod_kwargs
        # mu_e of incorrect size
        mod_kwargs['mu_e'] = make_nomask([self.dim + 2, 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_phi_e_size(self):
        """
        Tests whether size check asserts for phi_e is implemented correctly.
        """
        mod_kwargs = self.mod_kwargs
        # phi_e of incorrect size
        mod_kwargs['phi_e'] = make_nomask([self.dim + 2, self.dim - 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_sigma_e_size(self):
        """
        Tests whether size check asserts for sigma_e is implemented correctly.
        """
        mod_kwargs = self.mod_kwargs
        # sigma_e of incorrect size
        mod_kwargs['sigma_e'] = make_nomask([self.dim - 2, self.dim])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_var_data_nulls(self):
        """
        Tests whether if nulls appear in var_data AssertionError is raised.
        """
        mod_kwargs = self.mod_kwargs
        # replace a value in var_data with null
        mod_kwargs['var_data'][1, 1] = np.nan
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_yc_data_nulls(self):
        """
        Tests whether if nulls appear in yc_data AssertionError is raised.
        """
        mod_kwargs = self.mod_kwargs
        # replace a value in var_data with null
        mod_kwargs['yc_data'][1, 1] = np.nan
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_no_estimated_values(self):
        """
        Tests if AssertionError is raised if there are no masked values in the
        estimation arrays, implying no parameters to be estimated.
        """
        mod_kwargs = self.mod_kwargs
        # replace a value in var_data with null
        mod_kwargs['lam_0_e'] = make_nomask([self.dim, 1])
        mod_kwargs['lam_1_e'] = make_nomask([self.dim, self.dim])
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

    #@unittest.skip("Skipping")
    def test_score(self):
        """
        Tests if score is calculated.
        """
        self.affine_obj.score(self.guess_params)

    #@unittest.skip("Skipping")
    def test_hessian(self):
        """
        Tests if hessian is calculated.
        """
        self.affine_obj.hessian(self.guess_params)

    #@unittest.skip("Skipping")
    def test_std_errs(self):
        """
        Tests if standard errors are calculated.
        """
        self.affine_obj.std_errs(self.guess_params)

    def test_params_to_array(self):
        """
        Tests if params_to_array function works correctly, with and without
        returning masked arrays.
        """
        arrays_no_mask = self.affine_obj.params_to_array(self.guess_params)
        for arr in arrays_no_mask:
            self.assertIsInstance(arr, np.ndarray)
            self.assertNotIsInstance(arr, np.ma.core.MaskedArray)
        arrays_w_mask = self.affine_obj.params_to_array(self.guess_params,
                                                        return_mask=True)
        for arr in arrays_w_mask:
            self.assertIsInstance(arr, np.ma.core.MaskedArray)

    def test_params_to_array_zeromask(self):
        """
        Tests if params_to_array_zeromask function works correctly.
        """
        guess_params_arr = np.array(self.guess_params)
        neqs = self.affine_obj.neqs
        guess_params_arr[:neqs] = 0
        guess_params = guess_params_arr.tolist()
        guess_length = self.affine_obj._gen_guess_length()
        params_guesses = self.affine_obj.params_to_array_zeromask(guess_params)
        updated_guesses = params_guesses[-1]
        self.assertEqual(len(updated_guesses), len(guess_params) - neqs)

        #ensure that number of masked in first
        count_masked_new = ma.count_masked(params_guesses[0])
        count_masked_orig = ma.count_masked(self.affine_obj.lam_0_e)
        self.assertEqual(count_masked_new, count_masked_orig - neqs)

    def test_loglike(self):
        """
        Tests of loglikelihood is calculated.
        """
        self.affine_obj.loglike(self.guess_params)

    def test_gen_pred_coef(self):
        """
        Tests if Python-driven gen_pred_coef function runs.
        """
        params = self.affine_obj.params_to_array(self.guess_params)
        self.affine_obj.gen_pred_coef(*params)

    def test_opt_gen_pred_coef(self):
        """
        Tests if C-driven gen_pred_coef function runs.
        """
        params = self.affine_obj.params_to_array(self.guess_params)
        self.affine_obj.opt_gen_pred_coef(*params)

    def test_py_C_gen_pred_coef_equal(self):
        """
        Tests if the Python-driven and C-driven gen_pred_coef functions produce
        the same result, up to 1e-14.
        """
        params = self.affine_obj.params_to_array(self.guess_params)
        py_gpc = self.affine_obj.gen_pred_coef(*params)
        c_gpc = self.affine_obj.opt_gen_pred_coef(*params)
        for aix, array in enumerate(py_gpc):
            np.testing.assert_allclose(array, c_gpc[aix], rtol=1e-14)

    def test__solve_unobs(self):
        """
        Tests if the _solve_unobs function runs.
        """
        guess_params = self.guess_params
        param_arrays = self.affine_obj.params_to_array(guess_params)
        a_in, b_in = self.affine_obj.gen_pred_coef(*param_arrays)
        result = self.affine_obj._solve_unobs(a_in=a_in, b_in=b_in)

    def test__affine_pred(self):
        """
        Tests if the _affine_pred function runs.
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
        error.
        """
        no_err_mat, err_mat = self.affine_obj._gen_mat_list()
        self.assertEqual(no_err_mat, [2])
        self.assertEqual(err_mat, [1,3,4,5])

class TestEstimationMethods(TestCase):
    """
    Tests for solution methods
    """
    def setUp(self):

        ## Non-linear least squares
        np.random.seed(100)

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
        squares case without any latent factors.
        """
        guess_params = self.guess_params_nolat
        method = 'nls'
        solved = self.affine_obj_nolat.solve(guess_params, method=method,
                                             alg='newton')

    def test_solve_ml(self):
        """
        Tests whether or not basic solve is performed for direct maximum
        likelihood with a single latent factor.
        """
        guess_params = self.guess_params_lat
        method = 'ml'
        self.affine_obj_lat.solve(guess_params, method=method, alg='bfgs',
                                  xtol=0.1, ftol=0.1)

if __name__ == '__main__':
    unittest.main()
