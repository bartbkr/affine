"""
These are utilies used by the affine model class
"""

import numpy as np
import pandas as px
import pickle
import smtplib
import datetime as dt

from operator import itemgetter

def pickle_file(obj=None, name=None):
    """
    Pass name without .pkl extension
    """
    pkl_file = open(name + ".pkl", "wb")
    pickle.dump(obj, pkl_file)
    pkl_file.close()

def robust(mod_data, mod_yc_data, method=None, lam_0_g=None, lam_1_g=None,
        start_date=None, passwd=None):
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

    lam_0, lam_1, delta_1, phi, sig, a_solve, b_solve, lam_cov = out_bsr
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

    print "Send mail: woohoo!"

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
