#include <Python.h>
#include <arrayobject.h>

/* ==== Set up the methods table ====================== */
static PyMethodDef _C_extensionsMethods[] = {
    {"gen_pred_coef", gen_pred_coef, METH_VARARGS},
    {NULL, NULL}
}

/*  ==== Matrix sum function ===== */
void mat_sum(int x, int y, double arr1[x][y], double arr2[x][y], 
             double result[x][y]) {
    int dim_x, dim_y;
    for (dim_x = 0; dim_x < x; dim_x++) {
        for (dim_y = 0; dim_y < y; dim_y++) {
            result[dim_x][dim_y] = arr1[dim_x][dim_y] + arr2[dim_x][dim_y]
        }
    }
}

/*  ==== Matrix product functions ===== */
void mat_prodct(int row1, int col1, int row2, int col2, double arr1[row1][col1], 
                double arr2[row2][col2], double result[row1][col2]) {

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

/*  ==== Matrix product functions tpose second argument ===== */
void mat_prodct_tpose1(int row1, int col1, int row2, int col2, double arr1[row1][col1],
                double arr2[row2][col2], double result[row1][row2]) {

    int dim1_row, dim1_col, dim2_col;

    for (dim1_row = 0; dim1_row < row1; dim1_row++) {
        for (dim2_row = 0; dim2_row < row; dim2_row++) {
            double sum = 0;
            for (dim1_col = 0; dim1_col < col1; dim1_col++) {
                sum += arr1[dim1_row][dim1_col] * arr2[dim2_row][dim1_col];
            }
            result[dim1_row][dim2_row] = sum;
        }
    }
}

/*  ==== Matrix product functions tpose second argument ===== */
void mat_prodct_tpose1(int row1, int col1, int row2, int col2, double arr1[row1][col1],
                double arr2[row2][col2], double result[row1][row2]) {

    int dim1_row, dim1_col, dim2_col;

    for (dim1_row = 0; dim1_row < row1; dim1_row++) {
        for (dim2_row = 0; dim2_row < row; dim2_row++) {
            double sum = 0;
            for (dim1_col = 0; dim1_col < col1; dim1_col++) {
                sum += arr1[dim1_row][dim1_col] * arr2[dim2_row][dim1_col];
            }
            result[dim1_row][dim2_row] = sum;
        }
    }
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
        phi_x, phi_y, sigma_x, sigma_y, max_mth, mth;

    /* Parse input arguments to function */

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!i",
        &PyArray_Type, &lam_0, &PyArray_Type, &lam_1, &PyArray_Type, &delta_1,
        &PyArray_Type, &mu, &PyArray_Type, &phi, &PyArray_Type, &sigma, 
        *max_mth))
        return NULL;
    if (NULL == lam_0 || NULL == lam_1 || NULL == delta_1 || NULL == mu || 
        NULL == phi || NULL == sigma || NULL == max_mth) return NULL;

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

    /*  Create C arrays */

    lam_0_c = pymatrix_to_Carrayptrs(lam_0);
    lam_1_c = pymatrix_to_Carrayptrs(lam_1);
    delta_1_c = pymatrix_to_Carrayptrs(delta_1); 
    mu_c = pymatrix_to_Carrayptrs(mu); 
    phi_c = pymatrix_to_Carrayptrs(phi); 
    sigma_c = pymatrix_to_Carrayptrs(sigma);

    /*  Initialize collector arrays */

    double a_pre[max_mth];
    double b_pre[delta_1_x][max_mth];

    a_pre[0] = 0;
    for (i = 0; i < delta_1_x; i++) {
        b_pre[i][0] = delta_1_c[i]
    }

    for (mth = 0; mth < (max_mth - 1); mth++) {
        a_pre[mth+1] = a_pre[mth] + 

    }

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



