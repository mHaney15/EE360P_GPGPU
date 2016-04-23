kernel void Mat_Trans(global double* answer, global double* Mat, int w) {
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);
	if((i < w)&&(j < w)){
		answer[i*w + j] = Mat[j*w + i];
	}
}
