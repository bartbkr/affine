#include <Python.h>
#include <arrayobject.h>

/* ==== Set up the methods table ====================== */
static PyMethodDef _C_extensionsMethods[] = {
    {"gen_pred_coef", gen_pred_coef, METH_VARARGS},
    {NULL, NULL}
}

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 

void init_C_extensions()  {
    (void) Py_InitModule("_C_extensions", _C_extensionsMethods);
    import_array();
}

static PyObject * gen_pred_coef(PyObject *self, PyObject *args)  {
    PyArrayObject *lam_0, *lam_1, *delta_1, *mu, *phi, *sigma;
    int lam_0_x, lam_0_y, lam_1_x, lam_1_y, delta_1_x, delta_1_y, mu_x, mu_y,
        phi_x, phi_y, sigma_x, sigma_y;

    /* Parse input arguments to function */

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
        &PyArray_Type, &lam_0, &PyArray_Type, &lam_1, &PyArray_Type, &delta_1,
        &PyArray_Type, &mu, &PyArray_Type, &phi, &PyArray_Type, &sigma))
        return NULL;
    if (NULL == lam_0 || NULL == lam_1 || NULL == delta_1 || NULL == mu || 
        NULL == phi || NULL == sigma) return NULL;

    /* Get dimesions of all input arrays */

    lam_0_x=dims[0]=lam_0->dimensions[0];
    lam_0_y=dims[1]=lam_0->dimensions[1];
    lam_1_x=dims[0]=lam_1->dimensions[0];
    lam_1_y=dims[1]=lam_1->dimensions[1];
    delta_1_x=dims[0]=delta_1->dimensions[0];
    delta_1_y=dims[1]=delta_1->dimensions[1];
    mu_x=dims[0]=mu->dimensions[0];
    mu_y=dims[1]=mu->dimensions[1];
    phi_x=dims[0]=phi->dimensions[0];
    phi_y=dims[1]=phi->dimensions[1];
    sigma_x=dims[0]=sigma->dimensions[0];
    sigma_y=dims[1]=sigma->dimensions[1];

}

/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
    double **c, *a;
    int i,n,m;
    
    n=arrayin->dimensions[0];
    m=arrayin->dimensions[1];
    c=ptrvector(n);
    a=(double *) arrayin->data;  /* pointer to arrayin data as double */
    for ( i=0; i<n; i++)  {
        c[i]=a+i*m;  }
    return c;
}

/* ==== Dot Product function ==== */

void *matmult(PyArray_Object *prod)  {
    PyArray_Object 
    double 
}



