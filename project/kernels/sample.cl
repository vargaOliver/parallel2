int newRandom2(unsigned long seedA, unsigned long seedB)
{
	unsigned int seed = seedA + 1;
	unsigned int t = seed ^ (seed << 11);  
	unsigned int result = seedB ^ (seedB >> 19) ^ (t ^ (t >> 8));
	return abs(result);
}


__kernel void hello_kernel(__global int* input1, __global int* output, __global atomic_int* finished)
	{
		int i = 0;
		int temp[ARRAY_SIZE];

		for (i = 0; i < ARRAY_SIZE; i++) {
			temp[i] = input1[i];
		}
		
		int randomindex1 = 0;
		int randomindex2 = 0;
		int sorted = 0;
		int attempt = 0;
		int temp_element = 0;

		do {
			randomindex1 = newRandom2(attempt, attempt) % ARRAY_SIZE;
			randomindex2 = newRandom2(attempt, randomindex1) % ARRAY_SIZE;
			
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
			
			attempt++;
		} while (sorted == 0 && atomic_load(finished) == 0);
		
		if (atomic_load(finished) == 0) {
			for (i = 0; i < ARRAY_SIZE; i++) {
				output[i] = temp[i];
			}
			atomic_store(finished, 1);
		}
    }