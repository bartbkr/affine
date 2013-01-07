#include "Python.h"
#include "arrayobject.h"
#include "C_extensions.h"
#include <math.h>

/* === Constants used in rest of program === */
const double half = 0.5;

/* ==== Set up the methods table ====================== */
static PyMethodDef _C_extensionsMethods[] = {
    {"gen_pred_coef", gen_pred_coef, METH_VARARGS},
    {NULL, NULL}
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 

void init_C_extensions()  {
    (void) Py_InitModule("_C_extensions", _C_extensionsMethods);
    import_array();
}

/*  Array helper functions */
/*  ==== Matrix sum function ===== */
void mat_sum(int rows, int cols, double arr1[rows][cols], 
             double arr2[rows][cols], double result[rows][cols]) {
    int dim_rows, dim_cols;
    for (dim_rows = 0; dim_rows < rows; dim_rows++) {
        for (dim_cols = 0; dim_cols < cols; dim_cols++) {
            result[dim_rows][dim_cols] = arr1[dim_rows][dim_cols]
                                    + arr2[dim_rows][dim_cols];
        }
    }
}

/*  ==== Matrix subtraction function ===== */
void mat_subtract(int rows, int cols, double arr1[rows][cols], 
                  double arr2[rows][cols], double result[rows][cols]) {
    int d_row, d_col;
    for (d_row = 0; d_row < rows; d_row++) {
        for (d_col = 0; d_col < cols; d_col++) {
            result[d_row][d_col] = arr1[d_row][d_col] - arr2[d_row][d_col];
        }
    }
}

/*  ==== Matrix product functions ===== */
void mat_prodct(int row1, int col1, double arr1[row1][col1], int row2, 
                int col2, double **arr2[row2][col2], 
                double result[row1][col2]) {

    /* What about case when results in single number */

    int dim1_row, dim1_col, dim2_col;

    for (dim1_row = 0; dim1_row < row1; dim1_row++) {
        for (dim2_col = 0; dim2_col < col2; dim2_col++) {
            double sum = 0;
            for (dim1_col = 0; dim1_col < col1; dim1_col++) {
                sum += arr1[dim1_row][dim1_col] * arr2[dim1_col][dim2_col];
            }
            result[dim1_row][dim2_col] = sum;
        }
    }
}

/*  ==== Matrix product functions tpose first argument ===== */
void mat_prodct_tpose1(int row1, int col1, double arr1[row1][col1], 
                       int row2, int col2, double arr2[row2][col2], 
                       double result[col1][col2]) {

    int dim1_row, dim1_col, dim2_col;

    for (dim1_col = 0; dim1_col < col1; dim1_col++) {
        for (dim2_col = 0; dim2_col < col2; dim2_col++) {
            double sum = 0;
            for (dim1_row = 0; dim1_row < row1; dim1_row++) {
                sum += arr1[dim1_row][dim1_col] * arr2[dim1_row][dim2_col];
            }
            result[dim1_col][dim2_col] = sum;
        }
    }
}

/*  ==== Matrix product functions tpose second argument ===== */
void mat_prodct_tpose2(int row1, int col1, double arr1[row1][col1],
                       int row2, int col2, double arr2[row2][col2], 
                       double result[row1][row2]) {

    int dim1_row, dim1_col, dim2_row;

    for (dim1_row = 0; dim1_row < row1; dim1_row++) {
        for (dim2_row = 0; dim2_row < row2; dim2_row++) {
            double sum = 0;
            for (dim1_col = 0; dim1_col < col1; dim1_col++) {
                sum += arr1[dim1_row][dim1_col] * arr2[dim2_row][dim1_col];
            }
            result[dim1_row][dim2_row] = sum;
        }
    }
}

static PyObject *gen_pred_coef(PyObject *self, PyObject *args)  {
    PyArrayObject *lam_0, *lam_1, *delta_1, *mu, *phi, *sigma, *a_fin_array,
                  *b_fin_array;
    int lam_0_rows, lam_0_cols, lam_1_rows, lam_1_cols, delta_1_rows,
        delta_1_cols, mu_rows, mu_cols, phi_rows, phi_cols, sigma_rows,
        sigma_cols, delta_0, max_mth, mth, next_mth, i;

    double **lam_0_c, **lam_1_c, **delta_1_c, **mu_c, **phi_c, **sigma_c,
           **a_fin, **b_fin;

    /* Parse input arguments to function */

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!i",
        &PyArray_Type, &lam_0, &PyArray_Type, &lam_1, &PyArray_Type, &delta_1,
        &PyArray_Type, &mu, &PyArray_Type, &phi, &PyArray_Type, &sigma, 
        &delta_0, &max_mth))
        return NULL;
    if (NULL == lam_0 || NULL == lam_1 || NULL == delta_1 || NULL == mu || 
        NULL == phi || NULL == sigma) return NULL;

    /* Get dimesions of all input arrays */

    lam_0_rows=lam_0->dimensions[0];
    lam_0_cols=lam_0->dimensions[1];
    lam_1_rows=lam_1->dimensions[0];
    lam_1_cols=lam_1->dimensions[1];
    delta_1_rows=delta_1->dimensions[0];
    delta_1_cols=delta_1->dimensions[1];
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
    delta_1_c = pymatrix_to_Carrayptrs(delta_1); 
    mu_c = pymatrix_to_Carrayptrs(mu); 
    phi_c = pymatrix_to_Carrayptrs(phi); 
    sigma_c = pymatrix_to_Carrayptrs(sigma);

    /*  Initialize collector arrays */
    int a_dims[2] = {max_mth, 1};
    int b_dims[2] = {delta_1_rows, max_mth};
    int b_pre_rows = delta_1_rows;

    a_fin_array = (PyArrayObject *) PyArray_FromDims(2, a_dims, NPY_DOUBLE);
    b_fin_array = (PyArrayObject *) PyArray_FromDims(2, b_dims, NPY_DOUBLE);

    double a_pre[max_mth];
    double b_pre[delta_1_rows][max_mth];

    a_fin = pymatrix_to_Carrayptrs(a_fin_array);
    b_fin = pymatrix_to_Carrayptrs(b_fin_array);

    /* Initialize intermediate arrays */
    /*  Elements for a_pre calculation */
    double dot_sig_lam_0[sigma_rows][lam_0_cols];
    double diff_mu_sigl[mu_rows][1];
    double dot_bpre_mu_sig1[1][1];

    double dot_b_pre_sig[1][sigma_cols];
    double dot_b_sigt[1][sigma_rows];
    double dot_b_sst_bt[1][1];

    /*  Elements for b_pre calculation */
    double dot_sig_lam_1[sigma_rows][lam_1_cols];
    double diff_phi_sig[phi_rows][phi_cols];
    double dot_phisig_b[phi_cols][1];
    
    /*  Perform operations */

    a_pre[0] = delta_0;
    a_fin[0][1] = -a_pre[0];
    for (i = 0; i < delta_1_rows; i++) {
        b_pre[i][0] = delta_1_c[i][1];
        b_fin[i][0] = -b_pre[i][0];
    }

    double b_pre_mth[delta_1_rows];

    for (mth = 0; mth < (max_mth - 1); mth++) {

        next_mth = mth + 1;

        for (i = 0; i < delta_1_rows; i++) {
            b_pre_mth[i] = b_pre[i][mth];
        }

        /* Calculate next a_pre element*/
        mat_prodct(sigma_rows, sigma_cols, sigma_c, 
                   lam_0_rows, lam_0_cols, lam_0_c,
                   dot_sig_lam_0);
        mat_subtract(mu_rows, mu_cols, mu_c, dot_sig_lam_0, diff_mu_sigl);
        mat_prodct_tpose1(b_pre_rows, 1, b_pre_mth, 
                          mu_rows, 1, diff_mu_sigl, 
                          dot_bpre_mu_sig1);

        mat_prodct_tpose1(b_pre_rows, 1, b_pre_mth,
                          sigma_rows, sigma_cols, sigma_c, 
                          dot_b_pre_sig);
        mat_prodct_tpose2(1, sigma_cols, dot_b_pre_sig,
                          sigma_rows, sigma_cols, sigma_c,
                          dot_b_sigt);
        mat_prodct(1, sigma_rows, dot_b_sigt,
                   b_pre_rows, 1, b_pre_mth,
                   dot_b_sst_bt);

        a_pre[next_mth] = a_pre[mth] +  dot_bpre_mu_sig1[1][1] + 
                        (half * dot_b_sst_bt[1][1]) - delta_0;
        a_fin[next_mth] = -a_pre[next_mth] / next_mth;

        /* Calculate next b_pre element */
        mat_prodct(sigma_rows, sigma_cols, sigma_c,
                   lam_1_rows, lam_1_cols, lam_1_c, 
                   dot_sig_lam_1);
        mat_subtract(phi_rows, phi_cols, phi_c, dot_sig_lam_1, diff_phi_sig);
        mat_prodct_tpose1(phi_rows, phi_cols, diff_phi_sig,
                          b_pre_rows, 1, b_pre_mth,
                          dot_phisig_b);

        for (i = 0; i < delta_1_rows; i++) {
            b_pre[i][next_mth] = dot_phisig_b[i][1] - delta_1_c[i][1];
            b_fin[i][next_mth] = -(b_pre[i][next_mth] / (next_mth));
        }
    }

    free_Carrayptrs(lam_0_c);
    free_Carrayptrs(lam_1_c);
    free_Carrayptrs(delta_1_c);
    free_Carrayptrs(mu_c);
    free_Carrayptrs(phi_c);
    free_Carrayptrs(sigma_c);
    free_Carrayptrs(a_fin);
    free_Carrayptrs(b_fin);

    PyObject *Result = Py_BuildValue("OO", a_fin_array, b_fin_array);
    Py_DECREF(a_fin_array);
    Py_DECREF(b_fin_array);

    return Result;
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

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
    double **v;
    v=(double **)malloc((size_t) (n*sizeof(double)));
    if (!v)   {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);  }
    return v;
}

/* ==== Free a double *vector (vec of pointers) ========================== */ 
void free_Carrayptrs(double **v)  {
    free((char*) v);
}
