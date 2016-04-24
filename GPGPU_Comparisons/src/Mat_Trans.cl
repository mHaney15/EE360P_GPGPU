#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void Mat_Trans(global const double* Mat, global double* answer, int n) {
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	answer[i*n + j] = Mat[j*n + i];
}
