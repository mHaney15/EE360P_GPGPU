/* kernel.cl
 * Matrix multiplication: C = A * B.
 * Device code.
 */

// OpenCL Kernel
kernel void matrixMul(global int* answer, global int* A, global int* B, int wA, int wB)
{

   int tx = get_global_id(0);
   int ty = get_global_id(1);

   // value stores the element that is
   // computed by the thread
   int value = 0;
   for (int k = 0; k < wA; ++k)
   {
      int elementA = A[ty * wA + k];
      int elementB = B[k * wB + tx];
      value += elementA * elementB;
   }

   // Write the matrix to device memory each
   // thread writes one element
   C[ty * wA + tx] = value;
}

//source: http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl
