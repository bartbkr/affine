#include "Python.h"
#include "arrayobject.h"
#include "C_extensions.h"
#include <math.h>
#include <stdio.h>

struct module_state {
    PyObject *error;
};

/* === Constants used in rest of program === */
const double half = (double)1 / (double)2;


/* ==== Set up the methods table ====================== */
static PyMethodDef _C_extensions_methods[] = {
    {"gen_pred_coef", gen_pred_coef, METH_VARARGS, NULL},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

// All of this is specific code for Python 3
#if PY_MAJOR_VERSION >= 3

static int _C_extensions_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _C_extensions_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_C_extensions",
        NULL,
        sizeof(struct module_state),
        _C_extensions_methods,
        NULL,
        _C_extensions_traverse,
        _C_extensions_clear,
        NULL
};

/* ==== Initialize the _C_extensions functions ====================== */
// Module name must be _C_extensions in compile and linked
//

#define INITERROR return NULL

PyObject *
PyInit__C_extensions(void)

#else
#define INITERROR return

void
init_C_extensions(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_C_extensions", _C_extensions_methods);
#endif
    import_array();
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

/*  Array helper functions */
/*  ==== Matrix sum function ===== */
void mat_sum(int rows, int cols, double *arr1, double *arr2,
             double *result) {
    int mat_size, inc;
    mat_size = rows * cols;
    for (inc=0;inc < mat_size;inc++) {
        *result = *arr1 + *arr2;
        arr1++;
        arr2++;
        result++;
    }
}

/*  ==== Matrix subtraction function ===== */
void mat_subtract(int rows, int cols, double *arr1, double *arr2,
                  double *result) {
    int mat_size, inc;
    mat_size = rows * cols;
    for (inc=0;inc < mat_size;inc++) {
        *result = *arr1 - *arr2;
        arr1++;
        arr2++;
        result++;
    }
}

/*  ==== Matrix product functions ===== */
void mat_prodct(int row1, int col1, double *arr1,
                int col2, double *arr2,
                double *result) {

    /* What about case when results in single number */

    int dim1_row, dim2_col, dim1_col, col1_mod, row2_mod;
    double sum, *arr1pt, *arr2pt;

    col1_mod = 0;
    for (dim1_row = 0; dim1_row < row1; dim1_row++) {
        row2_mod = 0;
        for (dim2_col = 0; dim2_col < col2; dim2_col++) {
            arr1pt = &arr1[col1_mod];
            arr2pt = &arr2[row2_mod];
            sum = 0;
            for (dim1_col = 0; dim1_col < col1; dim1_col++) {
                sum += (*arr1pt) * (*arr2pt);
                arr1pt++;
                arr2pt+=col2;
            }
            *result = sum;
            result++;
            row2_mod++;
        }
        col1_mod += col1;
    }
}

/*  ==== Matrix product functions tpose first argument ===== */
void mat_prodct_tpose1(int row1, int col1, double *arr1,
                       int col2, double *arr2,
                       double *result) {

    int dim1_row, dim1_col, dim2_col, row1_mod, row2_mod;
    double sum, *arr1pt, *arr2pt;

    row1_mod = 0;
    for (dim1_col = 0;dim1_col < col1;dim1_col++) {
        row2_mod = 0;
        for (dim2_col = 0;dim2_col < col2;dim2_col++) {
            arr1pt = &arr1[row1_mod];
            arr2pt = &arr2[row2_mod];
            sum = 0;
            for (dim1_row = 0;dim1_row < row1;dim1_row++) {
                sum += (*arr1pt) * (*arr2pt);
                arr1pt+=col1;
                arr2pt+=col2;
            }
            *result = sum;
            result++;
            row2_mod++;
        }
        row1_mod++;
    }
}

/*  ==== Matrix product functions tpose second argument ===== */
void mat_prodct_tpose2(int row1, int col1, double *arr1,
                       int row2, double *arr2,
                       double *result) {

    int dim1_row, dim2_row, dim1_col, col1_mod, col2_mod;
    double sum, *arr1pt, *arr2pt;

    col1_mod = 0;
    for (dim1_row = 0; dim1_row < row1; dim1_row++) {
        col2_mod = 0;
        for (dim2_row = 0; dim2_row < row2; dim2_row++) {
            arr1pt = &arr1[col1_mod];
            arr2pt = &arr2[col2_mod];
            sum = 0;
            for (dim1_col = 0; dim1_col < col1; dim1_col++) {
                sum += (*arr1pt) * (*arr2pt);
                arr1pt++;
                arr2pt++;
            }
            *result = sum;
            result++;
            col2_mod += col1;
        }
        col1_mod += col1;
    }
}

static PyObject *gen_pred_coef(PyObject *self, PyObject *args)  {
    PyArrayObject *lam_0, *lam_1, *delta_0, *delta_1, *mu, *phi, *sigma,
                  *a_fin_array, *b_fin_array;

    int lam_0_cols, lam_1_cols, mu_rows, mu_cols, phi_rows, phi_cols,
        sigma_rows, sigma_cols, mat, bp_offset, bp_noffset, next_mat, i;

    const int max_mat;

    double *lam_0_c, *lam_1_c, *delta_0_c, *delta_1_c, *mu_c, *phi_c,
           *sigma_c, divisor;

    /* Parse input arguments to function */

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!i",
        &PyArray_Type, &lam_0, &PyArray_Type, &lam_1, &PyArray_Type, &delta_0,
        &PyArray_Type, &delta_1, &PyArray_Type, &mu, &PyArray_Type, &phi,
        &PyArray_Type, &sigma, &max_mat))
        return NULL;
    if (NULL == lam_0 || NULL == lam_1 || NULL == delta_0 || NULL == delta_1 ||
        NULL == mu || NULL == phi || NULL == sigma) return NULL;

    /* Get dimesions of all input arrays */

    lam_0_cols=lam_0->dimensions[1];
    lam_1_cols=lam_1->dimensions[1];
    const int delta_1_rows=delta_1->dimensions[0];
    mu_rows=mu->dimensions[0];
    mu_cols=mu->dimensions[1];
    phi_rows=phi->dimensions[0];
    phi_cols=phi->dimensions[1];
    sigma_rows=sigma->dimensions[0];
    sigma_cols=sigma->dimensions[1];

    /*  Create C arrays */
    /* Maybe should be constants??? */

    lam_0_c = pymatrix_to_Carrayptrs(lam_0);
    lam_1_c = pymatrix_to_Carrayptrs(lam_1);
    delta_0_c = pymatrix_to_Carrayptrs(delta_0);
    delta_1_c = pymatrix_to_Carrayptrs(delta_1);
    mu_c = pymatrix_to_Carrayptrs(mu);
    phi_c = pymatrix_to_Carrayptrs(phi);
    sigma_c = pymatrix_to_Carrayptrs(sigma);

    /*  Initialize collector arrays */
    npy_intp a_dims[2] = {max_mat, 1};
    npy_intp b_dims[2] = {max_mat, delta_1_rows};
    int b_pre_rows = delta_1_rows;

    double a_pre[max_mat];
    double b_pre[max_mat * delta_1_rows];
    double *a_fin = (double*) malloc(max_mat*sizeof(double));
    double *b_fin = (double*) malloc(max_mat * delta_1_rows * sizeof(double));

    if (a_fin==NULL) {
        printf("Failed to allocate memory for a_fin\n");
    }
    if (b_fin==NULL) {
        printf("Failed to allocate memory for b_fin\n");
    }

    /* Initialize intermediate arrays */
    /*  Elements for a_pre calculation */
    double dot_sig_lam_0_c[sigma_rows * lam_0_cols];
    double diff_mu_sigl_c[mu_rows];
    double dot_bpre_mu_sig1_c[1];

    double dot_b_pre_sig_c[sigma_cols];
    double dot_b_sigt_c[sigma_rows];
    double dot_b_sst_bt_c[1];

    /*  Elements for b_pre calculation */
    double dot_sig_lam_1_c[sigma_rows * lam_1_cols];
    double diff_phi_sig_c[phi_rows * phi_cols];
    double dot_phisig_b_c[phi_cols];

    /*  Perform operations */
    a_pre[0] = -delta_0_c[0];
    a_fin[0] = -a_pre[0];
    for (i = 0;i < delta_1_rows;i++) {
        b_pre[i] = -delta_1_c[i];
        b_fin[i] = -b_pre[i];
    }

    double b_pre_mat_c[b_pre_rows];

    /* Calculate unchanging elements*/
    /* Debugged this looks good */
    mat_prodct(sigma_rows, sigma_cols, sigma_c,
                lam_0_cols, lam_0_c,
                dot_sig_lam_0_c);
    /* Debugged this looks good */
    mat_subtract(mu_rows, mu_cols, mu_c, dot_sig_lam_0_c, diff_mu_sigl_c);

    for (mat = 0; mat < (max_mat - 1); mat++) {

        next_mat = mat + 1;

        //Setup indexes
        bp_offset = mat * delta_1_rows;
        bp_noffset = next_mat * delta_1_rows;

        /*  think need this b_pre_mat for proper array reading */
        for (i = 0; i < b_pre_rows; i++) {
            b_pre_mat_c[i] = b_pre[bp_offset + i];
        }

        /* Debugged this call, seems to be fine */
        mat_prodct_tpose1(b_pre_rows, 1, b_pre_mat_c,
                          1, diff_mu_sigl_c,
                          dot_bpre_mu_sig1_c);

        /* debugged this, it looks good */
        mat_prodct_tpose1(b_pre_rows, 1, b_pre_mat_c,
                          sigma_cols, sigma_c,
                          dot_b_pre_sig_c);
        /* debugged this call, looks good */
        mat_prodct_tpose2(1, sigma_cols, dot_b_pre_sig_c,
                          sigma_rows, sigma_c,
                          dot_b_sigt_c);
        mat_prodct(1, sigma_rows, dot_b_sigt_c,
                   1, b_pre_mat_c,
                   dot_b_sst_bt_c);

        //Divisor to prepare for b_fin calculation
        divisor = (double)1 / ((double)next_mat + (double)1);

        a_pre[next_mat] = a_pre[mat] + dot_bpre_mu_sig1_c[0] +
                        (half * dot_b_sst_bt_c[0]) - delta_0_c[0];
        a_fin[next_mat] = -a_pre[next_mat] * divisor;

        /* Calculate next b elements */
        mat_prodct(sigma_rows, sigma_cols, sigma_c,
                   lam_1_cols, lam_1_c,
                   dot_sig_lam_1_c);
        mat_subtract(phi_rows, phi_cols, phi_c, dot_sig_lam_1_c,
                     diff_phi_sig_c);
        mat_prodct_tpose1(phi_rows, phi_cols, diff_phi_sig_c,
                          1, b_pre_mat_c,
                          dot_phisig_b_c);


        //Issue seems to be that b_fin is not able to dynaically allocate these
        //doubles using both
        for (i = 0; i < delta_1_rows; i++) {
            b_pre[bp_noffset + i] = dot_phisig_b_c[i] - delta_1_c[i];
            b_fin[bp_noffset + i] = -b_pre[bp_noffset + i] * divisor;
        }
    }

    /* Free core arrays */
    free(lam_0_c);
    free(lam_1_c);
    free(delta_0_c);
    free(delta_1_c);
    free(mu_c);
    free(phi_c);
    free(sigma_c);

    a_fin_array = (PyArrayObject *) PyArray_SimpleNewFromData(2, a_dims,
                                                              NPY_DOUBLE,
                                                              a_fin);
    PyArray_UpdateFlags(a_fin_array, NPY_OWNDATA);
    b_fin_array = (PyArrayObject *) PyArray_SimpleNewFromData(2, b_dims,
                                                              NPY_DOUBLE,
                                                              b_fin);
    PyArray_UpdateFlags(b_fin_array, NPY_OWNDATA);

    PyObject *Result = Py_BuildValue("OO", a_fin_array, b_fin_array);

    return Result;
}

/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double *pymatrix_to_Carrayptrs(PyArrayObject *arrayin) {
    double *c, *a, *inc;
    int i, mat_size, n, m;

    n = arrayin->dimensions[0];
    m = arrayin->dimensions[1];
    mat_size = n * m;
    c = malloc(n * m * sizeof(*c));
    a = (double *) arrayin->data;
    inc = c;
    for (i=0;i < mat_size;i++) {
        *c = *a;
        c++;
        a++;
    }
    return inc;
}

/* ==== Free a double *vector (vec of pointers) ========================== */
void free_Carrayptrs(double **v, int rows)  {
    int i;
    for (i = 0; i < rows; i++) {
        free(*(v + i));
    }
    free(v);
}

void free_CarrayfPy(double **v)  {
    free((char*) v);
}
