#include <stdio.h>
#include <stdlib.h>

#define arraylength 8

int my_func_d(int a, int b, int *pointer) {
    int c = a + b;
    int d = b % a;
    printf("Here is c: %d\n", c);
    printf("Here is d: %d\n", d);
    printf("here is pointer ref value: %d\n", *pointer);
    return d;
}

void mat_prodct(int arr1_x, int arr1_y, int arr2_x, int arr2_y, 
                double arr1[], double arr2[], double result[]) {
    int dim1_x, dim1_y, dim2_y;

    for (dim1_x = 0; dim1_x < arr1_x; dim1_x++) {
        for (dim2_y = 0; dim2_y < arr2_y; dim2_y++) {
            double sum = 0;
            for (dim1_y = 0; dim1_y < arr1_y; dim1_y++) {
                sum += *arr1[dim1_x][dim1_y] * *arr2[dim1_y][dim2_y];
            }
            result[dim1_x][dim2_y] = sum;
        }
    }
}

int main()
{
    enum sz{S=0,L=-3,XL};
    int arr1_x = 3;
    int arr1_y = 4;
    int arr2_x = 4;
    int arr2_y = 2;
    double arr1[arr1_x][arr1_y];
    double arr2[arr2_x][arr2_y];
    double result[arr1_x][arr2_y];

    mat_prodct(arr1_x, arr1_y, arr2_x, arr2_y, &arr1, &arr2, &result);

    /*  
    double resltarray = (double*) malloc(sizeof(double));
    int i;
    int j;
    int i = 1;
    for (i = 0; i < arr1_x; i++) {
        for (j = 0; j < arr1_y; j++) {
            arr1[i][j] = i + j;
            printf("Argument: %f\n", arr1[i][j]); 
        }
    }

    for (i = 1; i < arraylength; i++){
        arr[i] = 3.0 * i;
        printf("Digit %f\n", arr[i]);
    }

    int m = 3;
    int n = 4;
    int hola = 201;
    int *pn = &hola;
    puts("Hello world");
    printf("location %d\n", *pn);
    int d = my_func_d(m, n, &hola);
    printf("Here is called d: d = %d\n", d);
    return 0;*/
}


/* 
int nmax = 20;

int main ( int argc , char **argv) 
{ int a=0, b=1, c, n, nmax =25; 
    printf ( "%3d: %d\n" ,1 ,a ); 
    printf ( "%3d: %d\n" ,2 ,b ); 
    for (n =3; n<= nmax; n++) {
        c=a+b; a=b; b=c;
        printf ( "%3d: %d\n" ,n, c );
    }
    return 0; 
}
*/
