/* Mat_Mult.cl
 * Matrix multiplication: C = A * B.
 * Device code.
 */

// OpenCL Kernel
kernel void matrixMul(global double* answer, global double* A, global double* B, int wA, int wB)
{

   unsigned int i = get_global_id(0);
   unsigned int j = get_global_id(1);

   // value stores the element that is
   // computed by the thread
   if((i < wA)&&(j < wB)){
	   double value = 0;
	   for (int k = 0; k < wA; k++)
	   {
		  double elementA = A[i * wA + k];
		  double elementB = B[k * wB + j];
		  value += elementA * elementB;
	   }

	   // Write the matrix to device memory each
	   // thread writes one element
	   C[i * wA + j] = value;
	}
}

// based off code found at http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl
