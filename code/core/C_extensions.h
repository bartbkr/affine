/* Header for C extensions */

/* .... Python callable functions ..................*/
static PyObject *gen_pred_coef(PyObject *self, PyObject *args);

/*  Matrix utility functions */

void mat_sum(int x, int y, double arr1[x][y], double arr2[x][y],
             double result[x][y]);
void mat_subtract(int x, int y, double arr1[x][y], double arr2[x][y],
             double result[x][y]);
void mat_prodct(int row1, int col1, double arr1[row1][col1], int row2,
                int col2, double arr2[row2][col2],
                double result[row1][col2]);
void mat_prodct_tpose1(int row1, int col1, double arr1[row1][col1],
                       int row2, int col2, double arr2[row2][col2],
                       double result[col1][col2]);
void mat_prodct_tpose2(int row1, int col1, double arr1[row1][col1],
                       int row2, int col2, double arr2[row2][col2],
                       double result[row1][row2]);

/* .... C matrix utility functions ..................*/
PyArrayObject *pymatrix(PyObject *objin);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_Carrayptrs(double **v);

