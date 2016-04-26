#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void Vec_Add(global const double* a, global const double* b, global double* answer) {
  unsigned int xid = get_global_id(0);
	  answer[xid] = a[xid] + b[xid];
}

//source: https://github.com/jeffheaton/opencl-hello-world/blob/master/src/main/resources/cl/sum.txt
