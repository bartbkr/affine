"""
These are utilies used by the affine model class
"""

import numpy as np
import numpy.ma as ma
import pandas as px
import datetime as dt

import pickle
import smtplib
import decorator
import sys

from operator import itemgetter
from numpy.linalg import LinAlgError
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS

def pickle_file(obj=None, name=None):
    """
    Pass name without .pkl extension
    """
    pkl_file = open(name + ".pkl", "wb")
    pickle.dump(obj, pkl_file)
    pkl_file.close()

def robust(mod_data, mod_yc_data, method=None, lam_0_g=None, lam_1_g=None):
    """
    Function to run model with guesses, also generating 
    method : string
        method to pass to Affine.solve()
    mod_data : pandas DataFrame 
        model data
    mod_yc_data : pandas DataFrame
        model yield curve data
    lam_0_g : array
        Guess for lambda 0
    lam_1_g : array
        Guess for lambda 1
    """
    from affine import Affine
        
    # subset to pre 2005
    mod_data = mod_data[:217]
    mod_yc_data = mod_yc_data[:214]

    #anl_mths, mth_only_data = proc_to_mth(mod_yc_data)
    bsr = Affine(yc_data = mod_yc_data, var_data = mod_data)
    neqs = bsr.neqs

    #test sum_sqr_pe
    if lam_0_g is None:
        lam_0_g = np.zeros([5*4, 1])
        lam_0_g[:neqs] = np.array([[-0.1], [0.1], [-0.1], [0.1], [-0.1]])

    #set seed for future repl

    if lam_1_g is None:
        lam_1_g = np.zeros([5*4, 5*4])
        for eqnumb in range(neqs):
            if eqnumb % 2 == 0:
                mult = 1
            else: 
                mult = -1
            guess = [[mult*-0.1], [mult*0.1], [mult*-0.1], [mult*0.1], \
                     [mult*-0.1]]
            lam_1_g[eqnumb, :neqs, None] = np.array([guess])*np.random.random()

    #generate a and b for no risk 
    #a_nrsk, b_nrsk = bsr.gen_pred_coef(lam_0_nr, lam_1_nr, bsr.delta_1,
                    #bsr.phi, bsr.sig)

    out_bsr = bsr.solve(lam_0_g, lam_1_g, method=method, ftol=1e-950,
                        xtol=1e-950, maxfev=1000000000, full_output=False)

    if method == "ls":
        lam_0, lam_1, delta_1, mu, phi, sig, a_solve, b_solve, output = out_bsr
        return lam_0, lam_1, output

    else:
        lam_0, lam_1, delta_1, mu, phi, sig, a_solve, b_solve, lam_cov = out_bsr
        return lam_0, lam_1, lam_cov

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

def flatten(array):
    """
    Flattens array to list values
    """
    a_list = []
    if array.ndim == 1:
        for index in range(np.shape(array)[0]):
            a_list.append(array[index])
        return a_list
    elif array.ndim == 2:
        rshape = np.reshape(array, np.size(array))
        for index in range(np.shape(rshape)[0]):
            a_list.append(rshape[index])
        return a_list
    
def select_rows(rows, array):
    """
    Creates 2-dim submatrix only of rows from list rows
    array must be 2-dim
    """
    if array.ndim == 1:
        new_array = array[rows[0]]
        if len(rows) > 1:
            for row in rows[1:]:
                new_array = np.append(new_array, array[row])
    elif array.ndim == 2:
        new_array = array[rows[0], :]
        if len(rows) > 1:
            for row in enumerate(rows[1:]):
                new_array = np.append(new_array, array[row, :], axis=0)
    return new_array

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
            mth_only[('l_tr_m' + str(mth))] = data[col]
    col_dict = dict([( mth_only.columns[x], mths[x]) for x in
                range(n_cols)])
    cols = np.asarray(sorted(col_dict.iteritems(),
                    key=itemgetter(1)))[:,0].tolist()
    mth_only = mth_only.reindex(columns = cols)
    mths.sort()
    return mth_only

def gen_guesses(neqs, k_ar, lat):
    """
    Generates Ang and Piazzesi guesses for matrices
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

def retry(func, attempts):
    """
    Decorator that attempts a function multiple times, even with exception
    """
    def inner_wrapper(*args, **kwargs):
        for attempt in xrange(attempts):
            try:
                return func(*args, **kwargs)
                break
            except LinAlgError:
                print "Trying again, maybe bad initial run"
                print "LinAlgError:", sys.exc_info()[0]
                continue
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise
    return inner_wrapper

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

def ap_constructor(neqs, k_ar, lat):
    """
    Contructor for ang and piazzesi model
    """

    masklower = np.mask_indices(lat, np.tril)
    lower_ind = np.tril_indecies(lat)
    for numb, element in enumerate(lower_ind):
        lower_ind[numb] = element + neqs + k_ar

    dim = neqs * k_ar + lat
    lam_0 = ma.zeros([dim, 1])
    lam_1 = ma.zeros([dim, dim])
    delta_1 = ma.zeros([dim, 1])
    delta_1[-lat:, ] = np.array([[-0.0001], [0.0000], [0.0001]])
    mu = ma.zeros([dim, 1])
    phi = ma.zeros([dim, dim])
    sigma = ma.zeros([dim, dim])
    if lat:
        sigma[-lat:, -lat:] = np.identity(lat)
        phi[-lat:, -lat:] = \
                np.random.random(lat*lat).reshape((lat, -1)) / 100000

    #mask values to be estimated
    lam_0[:neqs, 0] = ma.masked
    lam_0[-lat:, 0] = ma.masked

    lam_1[:neqs, :neqs] = ma.masked
    lam_1[:neqs, -lat:] = ma.masked
    lam_1[-lat:, :neqs] = ma.masked
    lam_1[-lat:, -lat:] = ma.masked

    delta_1[-lat:, 0] = ma.masked

    mu[-lat:, 0] = ma.masked

    phi[lower_ind] = ma.masked

    sigma[lower_ind] = ma.masked

    return lam_0, lam_1, delta_1, mu, phi, sigma

def pass_ols(var_data, freq, lat, k_ar, neqs, delta_1, mu, phi, sigma, 
             rf_rate):
    """
    Inserts estimated OLS parameters into appropriate matrices

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

    phi_ols = np.zeros([k_ar * neqs, k_ar * neqs])
    phi_ols[:neqs] = coefs[1:].T
    phi_ols[neqs:obs_var, :(k_ar - 1) * neqs] = np.identity((k_ar - 1) * neqs)

    sigma_ols = np.zeros([k_ar * neqs, k_ar * neqs])
    sigma_ols[:neqs, :neqs] = sigma_u
    sigma_ols[neqs:obs_var, neqs:obs_var] = np.identity((k_ar - 1) * neqs)
    sigma_ols = np.tril(sigma_ols)
    
    macro = var_data.copy()[k_ar - 1:]
    macro["constant"] = 1
    #we will want to change this next one once we make delta_1 uncontrained
    #(see top of ang and piazzesi page 759)
    delta_1[:neqs] = OLS(rf_rate,
                         macro).fit().params[1:].values[None].T
    mu[:neqs * k_ar, 0, None] = mu_ols[None]
    phi[:neqs * k_ar, :neqs * k_ar] = phi_ols[None]
    sigma[:neqs * k_ar, :neqs * k_ar] = sigma_ols[None]

    return delta_1, mu, phi, sigma
