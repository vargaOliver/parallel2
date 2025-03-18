__kernel void hello_kernel(__global int* input1, __global int* input2, __global int* output, int n) 
	{
       if (get_global_id(0) < n) 
	   {
			int n_sqrt = (int)sqrt((half)n);
			output[get_global_id(0)] = 0;
			for (int k = 0; k < n_sqrt; k++)
			{
				output[get_global_id(0)] = output[get_global_id(0)] + input1[get_global_id(0) / n_sqrt * n_sqrt + k] * input2[get_global_id(0) % n_sqrt + k * n_sqrt];
			}
       }
    }