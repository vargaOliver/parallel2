int newRandom2(unsigned long seedA, unsigned long seedB)
{
	unsigned int seed = seedA + 1; // + global_id
	unsigned int t = seed ^ (seed << 11);  
	unsigned int result = seedB ^ (seedB >> 19) ^ (t ^ (t >> 8));
	return abs(result);
}


__kernel void hello_kernel(__global const int* input1, __global int* output, volatile __global atomic_int* finished)
	{
		int i = 0;
		int temp[ARRAY_SIZE];
		int sum = 0;

		for (i = 0; i < ARRAY_SIZE; i++) {
			temp[i] = input1[i];
		}
		
		int randomindex1 = 0;
		int randomindex2 = 0;
		int sorted = 0;
		int attempt = 0;
		int temp_element = 0;
		int global_id = get_global_id(0);
		
		do {
			randomindex1 = abs(newRandom2(attempt, global_id)) % ARRAY_SIZE;
			randomindex2 = abs(newRandom2(attempt, randomindex1)) % ARRAY_SIZE;
			temp_element = temp[randomindex1];
			temp[randomindex1] = temp[randomindex2];
			temp[randomindex2] = temp_element;
			
			sorted = 1;
			
			for (i = 0; i < ARRAY_SIZE - 1; i++) {
				if (temp[i] > temp[i + 1]) {
					sorted = 0;
					break;
				}
			}
			
			if (sorted == 1) {
				atomic_store(finished, 1);
				output[ARRAY_SIZE] = output[ARRAY_SIZE] + 1;
			
				for (i = 0; i < ARRAY_SIZE; i++) {
					output[i] = temp[i];
				}
			}
			
			/*
			if (sorted == 1) {
				if (atomic_cmpxchg(finished, 0, 1) == 0) {
					output[ARRAY_SIZE] = 1;
					for (i = 0; i < ARRAY_SIZE; i++) {
						output[i] = temp[i];
					}
				}
			}
			*/
			
			attempt++;
		} while (sorted == 0 && atomic_load(finished) == 0);
    }