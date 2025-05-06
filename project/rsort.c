#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <CL/cl.h>

const int SAMPLE_SIZE = 100;
const int ARRAY_SIZE = 10;

int newRandom(unsigned long seed)
{
	seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
	unsigned int result = seed >> 16;
	return result;
}

int newRandom2(unsigned long seedA, unsigned long seedB)
{
	unsigned int seed = seedA + 1; // + global_id
	unsigned int t = seed ^ (seed << 11);  
	unsigned int result = seedB ^ (seedB >> 19) ^ (t ^ (t >> 8));
	return abs(result);
}

int* generateVector()
{
	int *toReturn = malloc(ARRAY_SIZE * sizeof(int));
	//int offset = rand() % 1000;
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		//toReturn[i] = rand() % 100;
		toReturn[i] = newRandom2(0, i) % 10;
	}
	return toReturn;
}

int main(void)
{
    int i;
	int j;
    cl_int err;
	int error_code;
	int sample_size_sqrt = (int)sqrt((double)SAMPLE_SIZE);
	srand(time(NULL));

    // Get platform
    cl_uint n_platforms;
	cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
		return 0;
	}

    // Get device
	cl_device_id device_id;
	cl_uint n_devices;
	err = clGetDeviceIDs(
		platform_id,
		CL_DEVICE_TYPE_GPU,
		1,
		&device_id,
		&n_devices
	);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
		return 0;
	}

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
	const char* kernel_code = load_kernel_source("kernels/sample.cl", &error_code);
    if (error_code != 0) {
        printf("Source code loading error!\n");
        return 0;
    }
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
	char options[100];
	sprintf(options, "-D ARRAY_SIZE=%d", ARRAY_SIZE);
    err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        size_t real_size;
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &real_size
        );
        char* build_log = (char*)malloc(sizeof(char) * (real_size + 1));
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            real_size + 1,
            build_log,
            &real_size
        );
        // build_log[real_size] = 0;
        printf("Real size : %d\n", real_size);
        printf("Build log : %s\n", build_log);
        free(build_log);
        return 0;
    }
    cl_kernel kernel = clCreateKernel(program, "hello_kernel", NULL);

    // Create the host buffer and initialize it
    int* host_bufferA = (int*)malloc(ARRAY_SIZE * sizeof(int));
	int* host_bufferB = (int*)malloc(ARRAY_SIZE * sizeof(int));
	int* host_bufferC = (int*)malloc(ARRAY_SIZE * sizeof(int));
	
	int *vectorA = generateVector();
	int *vectorC = generateVector();
	
    for (i = 0; i < ARRAY_SIZE; ++i) {
        host_bufferA[i] = vectorA[i];
		host_bufferC[i] = vectorC[i];
    }
	for (i = 0; i < ARRAY_SIZE; i+=1) {
		host_bufferB[i] = 0;
		printf("%d, ", host_bufferC[i]);
	}
	printf("\n\n");

    // Create the device buffers
    cl_mem device_bufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, ARRAY_SIZE * sizeof(int), NULL, NULL);
	cl_mem device_bufferB = clCreateBuffer(context, CL_MEM_READ_WRITE, ARRAY_SIZE * sizeof(int), NULL, NULL);

    // Set kernel arguments
	i = 0;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_bufferA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_bufferB);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&i);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, NULL, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_bufferA,
        CL_FALSE,
        0,
        ARRAY_SIZE * sizeof(int),
        host_bufferA,
        0,
        NULL,
        NULL
    );
	
	clEnqueueWriteBuffer(
        command_queue,
        device_bufferB,
        CL_FALSE,
        0,
        ARRAY_SIZE * sizeof(int),
        host_bufferB,
        0,
        NULL,
        NULL
    );
	
    // Size specification
    size_t local_work_size[1] = {1};
    //size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
	size_t global_work_size[1] = {SAMPLE_SIZE};
	
	clock_t t0 = clock();
	
    // Apply the kernel on the range
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        global_work_size,
        local_work_size,
        0,
        NULL,
        NULL
    );
	
	clFinish(command_queue);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_bufferB,
        CL_TRUE,
        0,
        ARRAY_SIZE * sizeof(int),
        host_bufferB,
        0,
        NULL,
        NULL
    );
	clock_t t1 = clock();
	double time = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
	printf("Parallel time: %f sec\n\n", time);
	for (i = 0; i < ARRAY_SIZE; i+=1) {
		printf("%d, ", host_bufferB[i]);
	}
	printf("\n\n");
	
	t0 = clock();
	int *temp = (int*)malloc(ARRAY_SIZE * sizeof(int));

    for (i = 0; i < ARRAY_SIZE; i++) {
		temp[i] = host_bufferC[i];
    }
	int randomindex1 = 0;
	int randomindex2 = 0;
	int sorted = 0;
	int attempt = 0;
	int temp_element = 0;

	do {
		randomindex1 = rand() % ARRAY_SIZE;
		randomindex2 = rand() % ARRAY_SIZE;
		
		temp_element = temp[randomindex1];
		temp[randomindex1] = temp[randomindex2];
		temp[randomindex2] = temp_element;
		
		sorted = 1;
		for (i = 0; i < ARRAY_SIZE - 1; i+=1) {
			if (temp[i] > temp[i + 1]) {
				sorted = 0;
				break;
			}
		}
		attempt++;
	} while (sorted == 0);
	
	t1 = clock();
	time = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
	printf("Sequential time: %f sec, number of attempts: %d\n", time, attempt);
	printf("\n");
	
	for (i = 0; i < ARRAY_SIZE; i+=1) {
		printf("%d, ", temp[i]);
	}
	printf("\n");
	
    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(host_bufferA);
	free(host_bufferB);
	free(host_bufferC);
}
