#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void Vec_Add(global const double* a, global const double* b, global double* answer, int n) {
  unsigned int xid = get_global_id(0);
  //if(xid < *n){
	  answer[xid] = a[xid] + b[xid];
  //}
}

//source: https://github.com/jeffheaton/opencl-hello-world/blob/master/src/main/resources/cl/sum.txt
