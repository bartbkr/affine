"""
These are utilies for creating model parameters
"""
import pickle

import datetime as dt
import numpy as np
import numpy.ma as ma
import pandas as px
import smtplib

from operator import itemgetter
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from affine.model.affine import Affine

def make_nomask(dim):
    marray = ma.zeros(dim)
    marray[:, :] = ma.masked
    marray[:, :] = ma.nomask
    return marray

def ap_constructor(neqs, k_ar, lat):
    """
    Contructor for ang and piazzesi model
    """
    lower_ind = list(np.tril_indices(lat))
    for numb, element in enumerate(lower_ind):
        lower_ind[numb] = element + neqs * k_ar

    dim = neqs * k_ar + lat
    lam_0 = ma.zeros([dim, 1])
    lam_1 = ma.zeros([dim, dim])
    delta_0 = ma.zeros([1, 1])
    delta_1 = ma.zeros([dim, 1])
    delta_1[-lat:, 0] = [-0.0001, 0.0000, 0.0001]
    mu = ma.zeros([dim, 1])
    phi = ma.zeros([dim, dim])
    sigma = ma.zeros([dim, dim])
    #if lat:
    #    sigma[-lat:, -lat:] = np.identity(lat)
    #    phi[-lat:, -lat:] = \
    #            np.random.random(lat*lat).reshape((lat, -1)) / 100000

    #mask values to be estimated
    lam_0[:neqs, 0] = ma.masked
    lam_0[-lat:, 0] = ma.masked

    lam_1[:neqs, :neqs] = ma.masked
    lam_1[:neqs, -lat:] = ma.masked
    lam_1[-lat:, :neqs] = ma.masked
    lam_1[-lat:, -lat:] = ma.masked

    delta_0[:, :] = ma.masked
    delta_0[:, :] = ma.nomask

    delta_1[-lat:, :] = ma.masked

    mu[-lat:, 0] = ma.masked

    phi[lower_ind] = ma.masked

    sigma[:, :] = ma.masked
    sigma[:, :] = ma.nomask
    sigma[-lat:, -lat:] = np.identity(lat)

    return lam_0, lam_1, delta_0, delta_1, mu, phi, sigma

def bsr_constructor(neqs, k_ar):
    """
    Constructor for BSR
    """
    dim = neqs * k_ar
    lam_0 = ma.zeros([dim, 1])
    lam_1 = ma.zeros([dim, dim])
    delta_0 = ma.zeros([1, 1])
    delta_1 = ma.zeros([dim, 1])
    mu = ma.zeros([dim, 1])
    phi = ma.zeros([dim, dim])
    sigma = ma.zeros([dim, dim])

    lam_0[:neqs, 0] = ma.masked
    lam_1[:neqs, :neqs] = ma.masked

    delta_0[:, :] = ma.masked
    delta_0[:, :] = ma.nomask

    delta_1[:, :] = ma.masked
    delta_1[:, :] = ma.nomask

    mu[:, :] = ma.masked
    mu[:, :] = ma.nomask

    phi[:, :] = ma.masked
    phi[:, :] = ma.nomask

    sigma[:, :] = ma.masked
    sigma[:, :] = ma.nomask

    return lam_0, lam_1, delta_0, delta_1, mu, phi, sigma

def pass_ols(var_data, freq, lat, k_ar, neqs, delta_0, delta_1, mu, phi, sigma,
             rf_rate=None):
    """
    Inserts estimated OLS parameters into appropriate matrices

    delta_0 : array (1, 1)
        guess for element of delta_0
    delta_1 : array (neqs * k_ar + lat, 1)
        guess for elements of delta_1
    mu : array (neqs * k_ar + lat, 1)
        guess for elements of mu
    phi : array (neqs * k_ar + lat, neqs * k_ar + lat)
        guess for elements of phi
    sig : array (neqs * k_ar + lat, neqs * k_ar + lat)
        guess for elements of sigma
    """
    var_fit = VAR(var_data, freq=freq).fit(maxlags=k_ar)

    coefs = var_fit.params.values
    sigma_u = var_fit.sigma_u

    obs_var = neqs * k_ar

    mu_ols = np.zeros([k_ar*neqs, 1])
    mu_ols[:neqs] = coefs[0, None].T

    #phi is unconstrained but only non-zero in macro porition for current
    #period elements
    phi_ols = np.zeros([k_ar * neqs, k_ar * neqs])
    phi_ols[:neqs] = coefs[1:].T
    phi_ols[neqs:obs_var, :(k_ar - 1) * neqs] = np.identity((k_ar - 1) * neqs)

    #macro portion of sigma is assumed lower triangular
    sigma_ols = np.zeros([k_ar * neqs, k_ar * neqs])
    sigma_ols[:neqs, :neqs] = sigma_u
    sigma_ols[neqs:obs_var, neqs:obs_var] = np.identity((k_ar - 1) * neqs)
    if lat:
        sigma_ols = np.tril(sigma_ols)

    if lat:
        macro = var_data.copy()
        macro["constant"] = 1
        #we will want to change this next one once we make delta_1 uncontrained
        #(see top of ang and piazzesi page 759)
        params = OLS(rf_rate, macro).fit().params
        delta_0[0, 0] = params[-1]
        delta_1[:neqs, 0] = params[:-1]

    mu[:neqs * k_ar, 0, None] = mu_ols[None]
    phi[:neqs * k_ar, :neqs * k_ar] = phi_ols[None]
    sigma[:neqs * k_ar, :neqs * k_ar] = sigma_ols[None]

    return delta_0, delta_1, mu, phi, sigma

def params_to_list(lam_0=None, lam_1=None, delta_1=None, mu=None,
                   phi=None, sigma=None, multistep=0):
    """
    Creates a single list of params from guess arrays that is passed into
    solver
    lam_0 : array (neqs * k_ar + lat, 1)
        guess for elements of lambda_0
    lam_1 : array (neqs * k_ar + lat, neqs * k_ar + lat)
        guess for elements of lambda_1
    delta_1 : array (neqs * k_ar + lat, 1)
        guess for elements of delta_1
    mu : array (neqs * k_ar + lat, 1)
        guess for elements of mu
    phi : array (neqs * k_ar + lat, neqs * k_ar + lat)
        guess for elements of phi
    sigma : array (neqs * k_ar + lat, neqs * k_ar + lat)
        guess for elements of sigma
    """
    #we will integrate standard assumptions
    #these could be changed later, but need to think of a standard way of
    #bring them in

    all_arrays = [lam_0_e, lam_1_e, delta_1_e, mu_e, phi_e, sigma_e]

    guess_list = []

    #for struct in all_arrays:
        #guess_list.append(struct[ma.getmask(struct))

    #we assume that those params corresponding to lags are set to zero
    if lat:
        #we are assuming independence between macro factors and latent
        #factors
        guess_list.append(flatten(lam_0[:neqs]))
        guess_list.append(flatten(lam_0[-lat:]))
        guess_list.append(flatten(lam_1[:neqs, :neqs]))
        guess_list.append(flatten(lam_1[:neqs, -lat:]))
        guess_list.append(flatten(lam_1[-lat:, :neqs]))
        guess_list.append(flatten(lam_1[-lat:, -lat:]))
        guess_list.append(flatten(delta_1[-lat:, 0]))
        guess_list.append(flatten(mu[-lat:, 0]))
        guess_list.append(flatten(phi[-lat:, -lat:]))
        guess_list.append(flatten(sigma[-lat:, -lat:]))
    else:
        guess_list.append(flatten(lam_0[:neqs]))
        guess_list.append(flatten(lam_1[:neqs, :neqs]))

    #flatten this list into one dimension
    flatg_list = [item for sublist in guess_list for item in sublist]
    return flatg_list
def gen_guesses(neqs, k_ar, lat):
    """
    Generates Ang and Piazzesi guesses for matrices
    This method is no longer relevant
    """
    dim = neqs * k_ar + lat
    lam_0 = np.zeros([dim, 1])
    lam_1 = np.zeros([dim, dim])
    delta_1 = np.zeros([dim, 1])
    delta_1[-lat:, ] = np.array([[-0.0001], [0.0000], [0.0001]])
    mu = np.zeros([dim, 1])
    phi = np.zeros([dim, dim])
    sigma = np.zeros([dim, dim])
    if lat:
        sigma[-lat:, -lat:] = np.identity(lat)
        phi[-lat:, -lat:] = \
                np.random.random(lat*lat).reshape((lat, -1)) / 100000
    return lam_0, lam_1, delta_1, mu, phi, sigma

def robust(mod_data, mod_yc_data, method=None):
    """
    Function to run model with guesses, also generating
    method : string
        method to pass to Affine.solve()
    mod_data : pandas DataFrame
        model data
    mod_yc_data : pandas DataFrame
        model yield curve data
    """
    # subset to pre 2005
    mod_data = mod_data[:217]
    mod_yc_data = mod_yc_data[:214]

    k_ar = 4
    neqs = 5
    lat = 0

    lam_0_e = ma.zeros((k_ar * neqs, 1))
    lam_0_e[:neqs] = ma.masked

    lam_1_e = ma.zeros((k_ar * neqs, k_ar * neqs))
    lam_1_e[:neqs, :neqs] = ma.masked

    delta_0_e = ma.zeros([1, 1])
    delta_0_e[:, :] = ma.masked
    delta_0_e[:, :] = ma.nomask

    delta_1_e = ma.zeros([k_ar * neqs, 1])
    delta_1_e[:, :] = ma.masked
    delta_1_e[:, :] = ma.nomask
    delta_1_e[np.argmax(mod_data.columns == 'fed_funds')] = 1

    var_fit = VAR(mod_data, freq="M").fit(maxlags=k_ar)

    coefs = var_fit.params.values
    sigma_u = var_fit.sigma_u
    obs_var = neqs * k_ar

    mu_e = ma.zeros([k_ar*neqs, 1])
    mu_e[:, :] = ma.masked
    mu_e[:, :] = ma.nomask
    mu_e[:neqs] = coefs[0, None].T

    phi_e = ma.zeros([k_ar * neqs, k_ar * neqs])
    phi_e[:, :] = ma.masked
    phi_e[:, :] = ma.nomask
    phi_e[:neqs] = coefs[1:].T
    phi_e[neqs:obs_var, :(k_ar - 1) * neqs] = np.identity((k_ar - 1) * neqs)

    sigma_e = ma.zeros([k_ar * neqs, k_ar * neqs])
    sigma_e[:, :] = ma.masked
    sigma_e[:, :] = ma.nomask
    sigma_e[:neqs, :neqs] = sigma_u
    sigma_e[neqs:obs_var, neqs:obs_var] = np.identity((k_ar - 1) * neqs)

    #anl_mths, mth_only_data = proc_to_mth(mod_yc_data)
    bsr = Affine(yc_data = mod_yc_data, var_data = mod_data, lam_0_e=lam_0_e,
                 lam_1_e=lam_1_e, delta_0_e=delta_0_e, delta_1_e=delta_1_e,
                 mu_e=mu_e, phi_e=phi_e, sigma_e=sigma_e)
    neqs = bsr.neqs

    guess_length = bsr.guess_length

    guess_params = [0.0000] * guess_length

    for numb, element in enumerate(guess_params[:30]):
        element = 0.0001
        guess_params[numb] = element * (np.random.random() - 0.5)

    out_bsr = bsr.solve(guess_params=guess_params, method=method, ftol=1e-950,
                        xtol=1e-950, maxfev=1000000000, full_output=False)

    if method == "ls":
        lam_0, lam_1, delta_1, mu, phi, sig, a_solve, b_solve, output = out_bsr
        return lam_0, lam_1, output

    else:
        lam_0, lam_1, delta_1, mu, phi, sig, a_solve, b_solve, lam_cov = out_bsr
        return lam_0, lam_1, lam_cov

def pickle_file(obj=None, name=None):
    """
    Pass name without .pkl extension
    """
    pkl_file = open(name + ".pkl", "wb")
    pickle.dump(obj, pkl_file)
    pkl_file.close()

def success_mail(passwd):
    """
    Function to run upon successful run
    """
    print "Trying to send email"
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login("bartbkr", passwd)

    # Send email
    senddate = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d')
    subject = "Your job has completed"
    head = "Date: %s\r\nFrom: %s\r\nTo: %s\r\nSubject: %s\r\nX-Mailer:" \
        "My-Mail\r\n\r\n"\
    % (senddate, "bartbkr@gmail.com", "barbkr@gmail.com", subject)
    msg = '''
    Job has completed '''

    server.sendmail("bartbkr@gmail.com", "bartbkr@gmail.com", head+msg)
    server.quit()

    print "Send mail: woohoo!"

def fail_mail(date, passwd):
    """
    Messages sent upon run fail
    """
    print "Trying to send email"
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login("bartbkr", passwd)

    # Send email
    date = dt.datetime.strftime(date, '%m/%d/%Y %I:%M:%S %p')
    senddate = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d')
    subject = "This run failed"
    head = "Date: %s\r\nFrom: %s\r\nTo: %s\r\nSubject: %s\r\nX-Mailer:" \
        "My-Mail\r\n\r\n"\
    % (senddate, "bartbkr@gmail.com", "barbkr@gmail.com", subject)
    msg = '''
    Hey buddy, the run you started %s failed '''\
    % (date)

    server.sendmail("bartbkr@gmail.com", "bartbkr@gmail.com", head+msg)
    server.quit()

    print "Send fail mail: woohoo!"

def to_mth(data):
    """
    This function transforms the yield curve data so that the names are all
    in months
    (not sure if this is necessary)
    """
    mths = []
    fnd = 0
    n_cols = len(data.columns)
    for col in data.columns:
        if 'm' in col:
            mths.append(int(col[6]))
            if fnd == 0:
                mth_only = px.DataFrame(data[col],
                        columns = [col],
                        index=data.index)
                fnd = 1
            else:
                mth_only[col] = data[col]
        elif 'y' in col:
            mth = int(col[6:])*12
            mths.append(mth)
            mth_only[('trcr_m' + str(mth))] = data[col]
    col_dict = dict([( mth_only.columns[x], mths[x]) for x in
                range(n_cols)])
    cols = np.asarray(sorted(col_dict.iteritems(),
                    key=itemgetter(1)))[:,0].tolist()
    mth_only = mth_only.reindex(columns = cols)
    mths.sort()
    return mth_only

