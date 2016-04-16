kernel void Vec_Add(global const int* a, global const int* b, global int* answer) {
  unsigned int xid = get_global_id(0);
  answer[xid] = a[xid] + b[xid];
}

//source: https://github.com/jeffheaton/opencl-hello-world/blob/master/src/main/resources/cl/sum.txt
