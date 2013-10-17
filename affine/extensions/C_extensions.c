#include "Python.h"
#include "arrayobject.h"
#include "C_extensions.h"
#include <math.h>

/* === Constants used in rest of program === */
const double half = 1.0/2.0;

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
void mat_sum(int rows, int cols, double **arr1, double **arr2, 
             double **result) {
    int dim_rows, dim_cols;
    for (dim_rows = 0; dim_rows < rows; dim_rows++) {
        for (dim_cols = 0; dim_cols < cols; dim_cols++) {
            result[dim_rows][dim_cols] = arr1[dim_rows][dim_cols]
                                    + arr2[dim_rows][dim_cols];
        }
    }
}

/*  ==== Matrix subtraction function ===== */
void mat_subtract(int rows, int cols, double **arr1, double **arr2, 
                  double **result) {
    int d_row, d_col;
    for (d_row = 0; d_row < rows; d_row++) {
        for (d_col = 0; d_col < cols; d_col++) {
            result[d_row][d_col] = arr1[d_row][d_col] - arr2[d_row][d_col];
        }
    }
}

/*  ==== Matrix product functions ===== */
void mat_prodct(int row1, int col1, double **arr1, 
                int col2, double **arr2, 
                double **result) {

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
void mat_prodct_tpose1(int row1, int col1, double **arr1, 
                       int col2, double **arr2, 
                       double **result) {

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
void mat_prodct_tpose2(int row1, int col1, double **arr1,
                       int row2, double **arr2, 
                       double **result) {

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
    PyArrayObject *lam_0, *lam_1, *delta_0, *delta_1, *mu, *phi, *sigma,
                  *a_fin_array, *b_fin_array, *a_solved, *b_solved, *a_pre_in, 
                  *b_pre_in, *checker1, *checker2, *checker3;

    int lam_0_cols, lam_1_cols, mu_rows, mu_cols, phi_rows,
        phi_cols, sigma_rows, sigma_cols, mth, next_mth, i, bp_offset,
        bp_noffset;

    const int max_mth;

    double **lam_0_c, **lam_1_c, **delta_0_c, **delta_1_c, **mu_c, **phi_c,
           **sigma_c, **a_fin, **b_fin, **dot_sig_lam_0_c, **diff_mu_sigl_c,
           **dot_bpre_mu_sig1_c, **dot_b_pre_sig_c, **dot_b_sigt_c,
           **dot_b_sst_bt_c, **dot_sig_lam_1_c, **diff_phi_sig_c,
           **dot_phisig_b_c, **b_pre_mth_c;

    double **a_solved_c, **b_solved_c, **a_pre_c, **b_pre_c, **checker1_c, 
           **checker2_c, **checker3_c, checker3_diff0, checker3_diff1,
           checker3_diff2, checker3_diff3, checker3_diff4, checker3_diff5,
           checker3_diff6, checker3_diff7, checker3_diff8, checker3_diff9,
           checker3_diff10, b_pre_check0, b_pre_check1, b_pre_check2,
           b_pre_check3, b_pre_check4, b_pre_check5, b_pre_check6,
           b_pre_check7, b_pre_check8, b_pre_check9, b_pre_check10,
           b_fin_check0, b_fin_check1, b_fin_check2, b_fin_check3,
           b_fin_check4, b_fin_check5, b_fin_check6, b_fin_check7,
           b_fin_check8, b_fin_check9, b_fin_check10, diff_a_pre,
           diff_a_fin, checker1_diff, checker2_diff;

    /* Parse input arguments to function */

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!iO!O!O!O!O!O!O!",
        &PyArray_Type, &lam_0, &PyArray_Type, &lam_1, &PyArray_Type, &delta_0,
        &PyArray_Type, &delta_1, &PyArray_Type, &mu, &PyArray_Type, &phi,
        &PyArray_Type, &sigma, &max_mth, &PyArray_Type, &a_solved,
        &PyArray_Type, &b_solved, &PyArray_Type, &a_pre_in, &PyArray_Type, 
        &b_pre_in, &PyArray_Type, &checker1, &PyArray_Type, &checker2,
        &PyArray_Type, &checker3))
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

    /* Debugging */
    a_solved_c = pymatrix_to_Carrayptrs(a_solved);
    b_solved_c = pymatrix_to_Carrayptrs(b_solved);
    a_pre_c = pymatrix_to_Carrayptrs(a_pre_in);
    b_pre_c = pymatrix_to_Carrayptrs(b_pre_in);
    checker1_c = pymatrix_to_Carrayptrs(checker1);
    checker2_c = pymatrix_to_Carrayptrs(checker2);
    checker3_c = pymatrix_to_Carrayptrs(checker3);

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
    int a_dims[2] = {max_mth, 1};
    int b_dims[2] = {max_mth, delta_1_rows};
    int b_pre_rows = delta_1_rows;

    a_fin_array = (PyArrayObject *) PyArray_FromDims(2, a_dims, NPY_DOUBLE);
    b_fin_array = (PyArrayObject *) PyArray_FromDims(2, b_dims, NPY_DOUBLE);

    double a_pre[max_mth];
    double b_pre[max_mth * delta_1_rows];

    a_fin = pymatrix_to_Carrayptrs(a_fin_array);
    b_fin = pymatrix_to_Carrayptrs(b_fin_array);

    /* Initialize intermediate arrays */
    /*  Elements for a_pre calculation */
    dot_sig_lam_0_c = twodim_to_point(sigma_rows, lam_0_cols);
    diff_mu_sigl_c = twodim_to_point(mu_rows, 1);
    dot_bpre_mu_sig1_c = twodim_to_point(1, 1);

    dot_b_pre_sig_c = twodim_to_point(1, sigma_cols);
    dot_b_sigt_c = twodim_to_point(1, sigma_rows);
    dot_b_sst_bt_c = twodim_to_point(1, 1);

    /*  Elements for b_pre calculation */
    dot_sig_lam_1_c = twodim_to_point(sigma_rows, lam_1_cols);
    diff_phi_sig_c = twodim_to_point(phi_rows, phi_cols);
    dot_phisig_b_c = twodim_to_point(phi_cols, 1);
    
    /*  Perform operations */

    a_pre[0] = -delta_0_c[0][0];
    a_fin[0][0] = -a_pre[0];
    for (i = 0; i < delta_1_rows; i++) {
        b_pre[i] = -delta_1_c[0][i];
        b_fin[0][i] = -b_pre[i];
    }

    b_pre_mth_c = twodim_to_point(b_pre_rows, 1);

    /* Calculate unchanging elements*/
    /* Debugged this looks good */
    mat_prodct(sigma_rows, sigma_cols, sigma_c, 
                lam_0_cols, lam_0_c,
                dot_sig_lam_0_c);
    /* Debugged this looks good */
    mat_subtract(mu_rows, mu_cols, mu_c, dot_sig_lam_0_c, diff_mu_sigl_c);

    for (mth = 0; mth < (max_mth - 1); mth++) {

        next_mth = mth + 1;

        //Setup indexes
        bp_offset = mth * delta_1_rows;
        bp_noffset = next_mth * delta_1_rows;

        /*  think need this b_pre_mth for proper array reading */
        for (i = 0; i < b_pre_rows; i++) {
            b_pre_mth_c[i][0] = b_pre[bp_offset + i];
        }

        /* Debugged this call, seems to be fine */
        mat_prodct_tpose1(b_pre_rows, 1, b_pre_mth_c, 
                          1, diff_mu_sigl_c, 
                          dot_bpre_mu_sig1_c);

        /* debugged this, it looks good */
        mat_prodct_tpose1(b_pre_rows, 1, b_pre_mth_c,
                          sigma_cols, sigma_c, 
                          dot_b_pre_sig_c);
        /* debugged this call, looks good */
        mat_prodct_tpose2(1, sigma_cols, dot_b_pre_sig_c,
                          sigma_rows, sigma_c,
                          dot_b_sigt_c);
        mat_prodct(1, sigma_rows, dot_b_sigt_c,
                   1, b_pre_mth_c,
                   dot_b_sst_bt_c);

        /* debugged here */
        a_pre[next_mth] = a_pre[mth] + dot_bpre_mu_sig1_c[0][0] +
                        (half * dot_b_sst_bt_c[0][0]) - delta_0_c[0][0];
        a_fin[next_mth][0] = -a_pre[next_mth] / (next_mth + 1);

        double diff_a_pre = a_pre[next_mth] - a_pre_c[next_mth][0];
        double diff_a_fin = a_fin[next_mth][0] - a_solved_c[0][next_mth];
        double checker1_diff = dot_bpre_mu_sig1_c[0][0] - checker1_c[mth][0];
        double checker2_diff = dot_b_sst_bt_c[0][0] - checker2_c[mth][0];

        /* Calculate next b elements */
        mat_prodct(sigma_rows, sigma_cols, sigma_c,
                   lam_1_cols, lam_1_c, 
                   dot_sig_lam_1_c);
        mat_subtract(phi_rows, phi_cols, phi_c, dot_sig_lam_1_c, 
                     diff_phi_sig_c);
        mat_prodct_tpose1(phi_rows, phi_cols, diff_phi_sig_c,
                          1, b_pre_mth_c,
                          dot_phisig_b_c);

        //Issue seems to be that b_fin is not able to dynaically allocate these
        //doubles using both
        for (i = 0; i < delta_1_rows; i++) {
            b_pre[bp_noffset + i] = dot_phisig_b_c[i][0] - delta_1_c[0][i];

            b_fin[next_mth][i] = -(b_pre[bp_noffset + i] * 
                                     (1 / (next_mth + 1.0)));
        }

        checker3_diff0 = dot_phisig_b_c[0][0] - checker3_c[mth][0];
        checker3_diff1 = dot_phisig_b_c[1][0] - checker3_c[mth][1];
        checker3_diff2 = dot_phisig_b_c[2][0] - checker3_c[mth][2];
        checker3_diff3 = dot_phisig_b_c[3][0] - checker3_c[mth][3];
        checker3_diff4 = dot_phisig_b_c[4][0] - checker3_c[mth][4];
        checker3_diff5 = dot_phisig_b_c[5][0] - checker3_c[mth][5];
        checker3_diff6 = dot_phisig_b_c[6][0] - checker3_c[mth][6];
        checker3_diff7 = dot_phisig_b_c[7][0] - checker3_c[mth][7];
        checker3_diff8 = dot_phisig_b_c[8][0] - checker3_c[mth][8];
        checker3_diff9 = dot_phisig_b_c[9][0] - checker3_c[mth][9];
        checker3_diff10 = dot_phisig_b_c[10][0] - checker3_c[mth][10];
        b_pre_check0 = b_pre[bp_noffset + 0] - b_pre_c[next_mth][0];
        b_pre_check1 = b_pre[bp_noffset + 1] - b_pre_c[next_mth][1];
        b_pre_check2 = b_pre[bp_noffset + 2] - b_pre_c[next_mth][2];
        b_pre_check3 = b_pre[bp_noffset + 3] - b_pre_c[next_mth][3];
        b_pre_check4 = b_pre[bp_noffset + 4] - b_pre_c[next_mth][4];
        b_pre_check5 = b_pre[bp_noffset + 5] - b_pre_c[next_mth][5];
        b_pre_check6 = b_pre[bp_noffset + 6] - b_pre_c[next_mth][6];
        b_pre_check7 = b_pre[bp_noffset + 7] - b_pre_c[next_mth][7];
        b_pre_check8 = b_pre[bp_noffset + 8] - b_pre_c[next_mth][8];
        b_pre_check9 = b_pre[bp_noffset + 9] - b_pre_c[next_mth][9];
        b_pre_check10 = b_pre[bp_noffset + 10] - b_pre_c[next_mth][10];
        b_fin_check0 = b_fin[next_mth][0] - b_solved_c[next_mth][0];
        b_fin_check1 = b_fin[next_mth][1] - b_solved_c[next_mth][1];
        b_fin_check2 = b_fin[next_mth][2] - b_solved_c[next_mth][2];
        b_fin_check3 = b_fin[next_mth][3] - b_solved_c[next_mth][3];
        b_fin_check4 = b_fin[next_mth][4] - b_solved_c[next_mth][4];
        b_fin_check5 = b_fin[next_mth][5] - b_solved_c[next_mth][5];
        b_fin_check6 = b_fin[next_mth][6] - b_solved_c[next_mth][6];
        b_fin_check7 = b_fin[next_mth][7] - b_solved_c[next_mth][7];
        b_fin_check8 = b_fin[next_mth][8] - b_solved_c[next_mth][8];
        b_fin_check9 = b_fin[next_mth][9] - b_solved_c[next_mth][9];
        b_fin_check10 = b_fin[next_mth][10] - b_solved_c[next_mth][10];
        double whatthe = 0;
    }

    free_Carrayptrs(lam_0_c);
    free_Carrayptrs(lam_1_c);
    free_Carrayptrs(delta_0_c);
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
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin) {
    double **c, *a;
    int i,n,m;
    
    n = arrayin->dimensions[0];
    m = arrayin->dimensions[1];
    c = ptrvector(n);
    a = (double *) arrayin->data;  /* pointer to arrayin data as double */
    for ( i=0; i<n; i++)  {
        c[i] = a + i * m;  
    }
    return c;
}

// Setup function
double **pymatrix_to_Carray(PyArrayObject *arrayin) {
    double **c, *a;
    int i, rows, cols;
    //what i need to do here is create a series of pointers to both dimensions
    //of pyarray
    
    rows = arrayin->dimensions[0];
    cols = arrayin->dimensions[1];
    c = ptrvector(rows);
    a = (double *) arrayin->data;  /* pointer to arrayin data as double */
    for ( i=0; i<rows; i++)  {
        c[i] = a + i * cols;  
    }
    return c;
}

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n) {
    double **v;
    v=(double **)malloc((n*sizeof(double)));
    if (!v)   {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);  }
    return v;
}

/*  ==== Create ** double from double 2-dim array === */
double **twodim_to_point(int rows, int cols) {
    int row;
    double **pointer = malloc(rows * sizeof(double *));
    pointer[0] = malloc(rows * cols * sizeof(double));
    for(row = 1; row < rows; row++) {
        pointer[row] = pointer[0] + row * cols;
    }
    return pointer;
}


/* ==== Free a double *vector (vec of pointers) ========================== */ 
void free_Carrayptrs(double **v)  {
    free((char*) v);
}
