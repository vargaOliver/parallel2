__kernel void hello_kernel(__global int* input1, __global int* output, int n) 
	{
       if (get_global_id(0) < n) 
	   {
			output[get_global_id(0)] = 0;
       }
    }