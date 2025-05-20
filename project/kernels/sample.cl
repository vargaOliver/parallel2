int newRandom2(unsigned long seedA, unsigned long seedB)
{
	unsigned int seed = seedA + 1; // + global_id
	unsigned int t = seed ^ (seed << 11);  
	unsigned int result = seedB ^ (seedB >> 19) ^ (t ^ (t >> 8));
	return abs(result);
}


__kernel void hello_kernel(__global const int* input1, __global int* output, __global int* finished)
	{
		int i = 0;
		int temp[ARRAY_SIZE];
		int sum = 0;

		for (i = 0; i < ARRAY_SIZE; i++) {
			temp[i] = input1[i];
		}
		
		//*finished = 0;
		
		int randomindex1 = 0;
		int randomindex2 = 0;
		int sorted = 0;
		int attempt = 0;
		int temp_element = 0;
		int global_id = get_global_id(0);
		
		do {
			randomindex1 = abs(newRandom2(attempt, global_id)) % ARRAY_SIZE;
			randomindex2 = abs(newRandom2(attempt, randomindex1)) % ARRAY_SIZE;
			/*
			if (randomindex1 < 0 || randomindex1 >= ARRAY_SIZE) {
				output[0] = 1234;
				output[1] = randomindex1;
				output[2] = global_id;
				output[3] = attempt;
				return;
			}
			*/
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
		} while (sorted == 0);
		
		//*finished = 1;
		
		for (i = 0; i < ARRAY_SIZE; i++) {
			output[i] = temp[i];
		}

    }