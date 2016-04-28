/* Mat_Mult.cl
 * Matrix multiplication: C = A * B.
 * Device code.
 */

// OpenCL Kernel
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void Mat_Mult(global const double* a, global const double* b, global double* answer, int n)
{
   unsigned int i = get_global_id(0);
   unsigned int j = get_global_id(1);
   // value stores the element that is computed by the thread
   double value = n;
   for (int k = 0; k < n; k++)
   {
	   double elementA = a[(i*n) + k];
	   double elementB = b[(k*n) + j];
	   value += elementA * elementB;
   }
   // Write the matrix to device memory each thread writes one element
   answer[(i*n) + j] = value;
}
// based off code found at http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl
