import numpy as np
import pickle
import smtplib

import datetime as dt

from affine import Affine

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
        
    # subset to pre 2005
    mod_data = mod_data[:217]
    mod_yc_data = mod_yc_data[:214]

    #anl_mths, mth_only_data = proc_to_mth(mod_yc_data)
    bsr = Affine(yc_data = mod_yc_data, var_data = mod_data)
    neqs = bsr.neqs
    k_ar = bsr.k_ar

    #test sum_sqr_pe
    if lam_0_g is None:
        lam_0_g = np.zeros([5*4, 1])
        lam_0_g[:neqs] = np.array([[-0.1],[0.1],[-0.1],[0.1],[-0.1]])

    #set seed for future repl

    if lam_1_g is None:
        lam_1_g = np.zeros([5*4, 5*4])
        for x in range(neqs):
            if x % 2 == 0:
                mult = 1
            else: 
                mult = -1
            guess = [[mult*-0.1],[mult*0.1],[mult*-0.1],[mult*0.1],[mult*-0.1]]
            lam_1_g[x, :neqs, None] = np.array([guess])*np.random.random()

    #generate a and b for no risk 
    #a_nrsk, b_nrsk = bsr.gen_pred_coef(lam_0_nr, lam_1_nr, bsr.delta_1,
                    #bsr.phi, bsr.sig)

    out_bsr = bsr.solve(lam_0_g, lam_1_g, method=method, ftol=1e-950,
                        xtol=1e-950, maxfev=1000000000, full_output=False)

    lam_0_n, lam_1_n, delta_1_n, phi_n, sig_n, a, b, lam_cov= out_bsr
    return lam_0_n, lam_1_n, lam_cov

def success_mail(passwd):
    print "Trying to send email"
    server=smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login("bartbkr", passwd)

    # Send email
    senddate=dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d')
    subject="Your job has completed"
    m="Date: %s\r\nFrom: %s\r\nTo: %s\r\nSubject: %s\r\nX-Mailer: My-Mail\r\n\r\n"\
    % (senddate, "bartbkr@gmail.com", "barbkr@gmail.com", subject)
    msg='''
    Job has completed '''

    server.sendmail("bartbkr@gmail.com", "bartbkr@gmail.com", m+msg)
    server.quit()

    print "Send mail: woohoo!"

def fail_mail(date, passwd):
    print "Trying to send email"
    server=smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login("bartbkr", passwd)

    # Send email
    date = dt.datetime.strftime(date, '%m/%d/%Y %I:%M:%S %p')
    senddate=dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d')
    subject="This run failed"
    m="Date: %s\r\nFrom: %s\r\nTo: %s\r\nSubject: %s\r\nX-Mailer: My-Mail\r\n\r\n"\
    % (senddate, "bartbkr@gmail.com", "barbkr@gmail.com", subject)
    msg='''
    Hey buddy, the run you started %s failed '''\
    % (date)

    server.sendmail("bartbkr@gmail.com", "bartbkr@gmail.com", m+msg)
    server.quit()

    print "Send mail: woohoo!"
