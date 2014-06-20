from unittest import TestCase

import unittest
import numpy as np
import numpy.ma as ma
import pandas as pa

from affine.constructors.helper import make_nomask
from affine.model.affine import Affine

import ipdb

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
        Tests whether correct setup successfully initializes model object
        """
        model = Affine(**self.mod_kwargs)
        self.assertIsInstance(model, Affine)

    def test_wrong_lam0_size(self):
        """
        Tests whether size check asserts for lam_0_e is implemented correctly
        """
        mod_kwargs = self.mod_kwargs
        # lam_0_e of incorrect size
        mod_kwargs['lam_0_e'] = make_nomask([self.dim - 1, 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_lam1_size(self):
        """
        Tests whether size check asserts for lam_1_e is implemented correctly
        """
        mod_kwargs = self.mod_kwargs
        # lam_1_e of incorrect size
        mod_kwargs['lam_1_e'] = make_nomask([self.dim - 1, self.dim + 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_delta_1_size(self):
        """
        Tests whether size check asserts for delta_1_e is implemented correctly
        """
        mod_kwargs = self.mod_kwargs
        # delta_1_e of incorrect size
        mod_kwargs['delta_1_e'] = make_nomask([self.dim + 1, 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_mu_e_size(self):
        """
        Tests whether size check asserts for mu_e is implemented correctly
        """
        mod_kwargs = self.mod_kwargs
        # mu_e of incorrect size
        mod_kwargs['mu_e'] = make_nomask([self.dim + 2, 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_phi_e_size(self):
        """
        Tests whether size check asserts for phi_e is implemented correctly
        """
        mod_kwargs = self.mod_kwargs
        # phi_e of incorrect size
        mod_kwargs['phi_e'] = make_nomask([self.dim + 2, self.dim - 1])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_wrong_sigma_e_size(self):
        """
        Tests whether size check asserts for sigma_e is implemented correctly
        """
        mod_kwargs = self.mod_kwargs
        # sigma_e of incorrect size
        mod_kwargs['sigma_e'] = make_nomask([self.dim - 2, self.dim])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_var_data_nulls(self):
        """
        Tests whether if nulls appear in var_data AssertionError is raised
        """
        mod_kwargs = self.mod_kwargs
        # replace a value in var_data with null
        mod_kwargs['var_data'][1, 1] = np.nan
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_yc_data_nulls(self):
        """
        Tests whether if nulls appear in yc_data AssertionError is raised
        """
        mod_kwargs = self.mod_kwargs
        # replace a value in var_data with null
        mod_kwargs['yc_data'][1, 1] = np.nan
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

    def test_no_estimated_values(self):
        """
        Tests if error is raised if there are no masked values in the
        estimation arrays, implying no parameters to be estimated
        """
        mod_kwargs = self.mod_kwargs
        # replace a value in var_data with null
        mod_kwargs['lam_0_e'] = make_nomask([self.dim, 1])
        mod_kwargs['lam_1_e'] = make_nomask([self.dim, self.dim])
        self.assertRaises(AssertionError, Affine, **mod_kwargs)

class TestSolveMethods(TestCase):
    """
    Tests for methods related to solve methods
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
        self.affine_object = Affine(**self.mod_kwargs)

    @unittest.skip("Skipping")
    def test_score(self):
        """
        Tests if score is calculated
        """
        self.affine_object.score(self.guess_params)

    @unittest.skip("Skipping")
    def test_hessian(self):
        """
        Tests if hessian is calculated
        """
        self.affine_object.hessian(self.guess_params)

    @unittest.skip("Skipping")
    def test_std_errs(self):
        """
        Tests if standard errors are calcualted
        """
        self.affine_object.std_errs(self.guess_params)

    def test_params_to_array(self):
        """
        Tests if params_to_array function works correctly, with and without
        returning mask
        """
        arrays_no_mask = self.affine_object.params_to_array(self.guess_params)
        for



    def test_loglike(self):
        """
        Tests of loglikelihood is calculated
        """
        self.affine_object.loglike(self.guess_params)

    def test_gen_pred_coef(self):
        """
        Tests if Python-driven gen_pred_coef function works
        """
        params = self.affine_object.params_to_array(self.guess_params)
        self.affine_object.gen_pred_coef(*params)

    def test_opt_gen_pred_coef(self):
        """
        Tests if C-driven gen_pred_coef function works
        """
        params = self.affine_object.params_to_array(self.guess_params)
        self.affine_object.opt_gen_pred_coef(*params)

    def test_py_C_gen_pred_coef_equal(self):
        """
        Tests if the Python-driven and C-driven gen_pred_coef functions produce
        the same result
        """
        params = self.affine_object.params_to_array(self.guess_params)
        py_gpc = self.affine_object.gen_pred_coef(*params)
        c_gpc = self.affine_object.opt_gen_pred_coef(*params)
        for aix, array in enumerate(py_gpc):
            np.testing.assert_allclose(array, c_gpc[aix], rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
