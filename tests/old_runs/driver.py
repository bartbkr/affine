"""
This simple program runs tp_proj.py and sends an email upon fail
"""
import keyring
import sys

import datetime as dt

from core.util import fail_mail

start_date = dt.datetime.now()

#get passwd from keyring
passwd = keyring.get_password("email_auth", "bartbkr") 

try:
    execfile("tp_proj_unob.py")
except:
    print "Error:", sys.exc_info()[0]
    fail_mail(start_date, passwd)
    raise
