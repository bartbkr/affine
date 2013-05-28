# coding: utf-8
get_ipython().magic(u'run tp_proj_unob.py')
solve
bsr_solve
solve = bsr_solve[-1]
solve
solve.params
solve.tvalues
bsr_model.hessian
bsr_model.hessian(params)
bsr_model.hessian(bsr_solve.params)
bsr_solve
params
params = bsr_solve.params
params = bsr_solve[-1].params
params
bsr_model.hessian(params)
bsr_model.hessian(params, args=(lam_0_g, lam_1_g, delta_1_g, mu_g, phi_g, sigma_g))
))
bsr_model.hessian(params, args=(lam_0_g, lam_1_g, delta_1_g, mu_g, phi_g, sigma_g))