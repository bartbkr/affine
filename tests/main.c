#include <stdio.h>

/*  ==== Matrix product functions ===== */
void mat_prodct(int row1, int col1, double *arr1, 
                int col2, double *arr2, 
                double *result) {

    /* What about case when results in single number */

    int inc, mat_size, dim1_row, dim2_col, dim1_col, col1_mod, row2_mod;
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

    int inc, mat_size, dim1_row, dim1_col, dim2_col, row1_mod, row2_mod;
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

    int inc, mat_size, dim1_row, dim2_row, dim1_col, col1_mod, col2_mod;
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

int main( int argc, const char* argv[] )
{
    int row1, col1, col2, iter, col3;
    row1 = 2;
    col1 = 3;
    col2 = 3;
    col3 = 4;

    double arr1[row1 * col1];
    double arr2[col1 * col2];
    double arr3[row1 * col1];
    double arr4[row1 * row1];
    double arr5[col3 * col1];
    double results[row1 * col2];
    double results2[row1 * col1];
    double results3[col1 * col2];
    double results4[row1 * col3];

    /* array 1 data */
    arr1[0] = 3;
    arr1[1] = 1;
    arr1[2] = 2;
    arr1[3] = 1;
    arr1[4] = 2;
    arr1[5] = 3;

    /* array 2 data */
    arr2[0] = 1;
    arr2[1] = 3;
    arr2[2] = 6;
    arr2[3] = 2;
    arr2[4] = 4;
    arr2[5] = 7;
    arr2[6] = 2;
    arr2[7] = 5;
    arr2[8] = 8;

    /* array 3 data */
    arr3[0] = 4;
    arr3[1] = 2;
    arr3[2] = 3;
    arr3[3] = 2;
    arr3[4] = 3;
    arr3[5] = 4;

    /* array 4 data */
    arr4[0] = 1;
    arr4[1] = 2;
    arr4[2] = 3;
    arr4[3] = 4;

    /* array 5 data */
    arr5[0] = 1;
    arr5[1] = 3;
    arr5[2] = 6;
    arr5[3] = 2;
    arr5[4] = 4;
    arr5[5] = 7;
    arr5[6] = 2;
    arr5[7] = 5;
    arr5[8] = 8;
    arr5[9] = 8;
    arr5[10] = 9;
    arr5[11] = 11;


    /*mat_prodct(row1, col1, arr1, col2, arr2, results);*/
    /*mat_prodct_tpose1(row1, col1, arr1, row1, arr4, results3);*/
    mat_prodct_tpose2(2, 3, arr1, 4, arr5, results4);
    /*mat_sum(row1, col1, arr1, arr3, results2);*/

    /*
    for (iter=0;iter < (row1 * col2);iter++) {
        printf("Element %f\n", results[iter]); 
    }
    */
    
    
    for (iter=0;iter < (2 * 4);iter++) {
        printf("Element %f\n", results4[iter]); 
    }

    /*
    for (iter=0;iter < (row1 * row1);iter++) {
        printf("Element %f\n", results2[row1]); 
    }
    */

}
