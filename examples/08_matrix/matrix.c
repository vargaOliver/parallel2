#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <CL/cl.h>

const int SAMPLE_SIZE = 1000000;
const int PRINT_STEP = 1000;

int* generateVector()
{
	int *toReturn = malloc(SAMPLE_SIZE * sizeof(int));
	for (int i = 0; i < SAMPLE_SIZE; ++i) {
		toReturn[i] = rand() % 10;
		//toReturn[i] = i + 1;
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
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
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
    int* host_bufferA = (int*)malloc(SAMPLE_SIZE * sizeof(int));
	int* host_bufferB = (int*)malloc(SAMPLE_SIZE * sizeof(int));
	int* host_bufferC = (int*)malloc(SAMPLE_SIZE * sizeof(int));
	int* host_bufferD = (int*)malloc(SAMPLE_SIZE * sizeof(int));
	
	int *vectorA = generateVector();
	int *vectorB = generateVector();
	int *vectorC = generateVector();
	
    for (i = 0; i < SAMPLE_SIZE; ++i) {
        host_bufferA[i] = vectorA[i];
		host_bufferB[i] = vectorB[i];
    }

    // Create the device buffers
    cl_mem device_bufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);
	cl_mem device_bufferB = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);
	cl_mem device_bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_bufferA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_bufferB);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&SAMPLE_SIZE);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, NULL, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_bufferA,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(int),
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
        SAMPLE_SIZE * sizeof(int),
        host_bufferB,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 512;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;
	
	clock_t t0 = clock();
	
    // Apply the kernel on the range
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
    );

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_bufferC,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(int),
        host_bufferC,
        0,
        NULL,
        NULL
    );
	clock_t t1 = clock();
	double time = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
	printf("Parallel time: %f sec\n", time);
	
	t0 = clock();
	for (i = 0; i < SAMPLE_SIZE; i+=1) {
		host_bufferD[i] = 0;
        for (j = 0; j < sample_size_sqrt; j+=1) {
			host_bufferD[i] += host_bufferA[i / sample_size_sqrt * sample_size_sqrt + j] * host_bufferB[i % sample_size_sqrt + j * sample_size_sqrt];
		}
	}
	t1 = clock();
	time = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
	printf("Sequential time: %f sec\n", time);
	/*
    for (i = 0; i < sample_size_sqrt; i+=1) {
        for (j = 0; j < sample_size_sqrt; j+=PRINT_STEP) {
			printf("A[%d] = %d, ", i * sample_size_sqrt + j, host_bufferA[i * sample_size_sqrt + j]);
		}
		printf("\n");
    }
	printf("\n");
	for (i = 0; i < sample_size_sqrt; i+=1) {
        for (j = 0; j < sample_size_sqrt; j+=PRINT_STEP) {
			printf("B[%d] = %d, ", i * sample_size_sqrt + j, host_bufferB[i * sample_size_sqrt + j]);
		}
		printf("\n");
    }
	*/
	printf("\n");
	for (i = 0; i < sample_size_sqrt; i+=1) {
        for (j = 0; j < sample_size_sqrt; j+=PRINT_STEP) {
			printf("C[%d] = %d, ", i * sample_size_sqrt + j, host_bufferC[i * sample_size_sqrt + j]);
		}
		printf("\n");
    }
	printf("\n");
	for (i = 0; i < sample_size_sqrt; i+=1) {
        for (j = 0; j < sample_size_sqrt; j+=PRINT_STEP) {
			printf("D[%d] = %d, ", i * sample_size_sqrt + j, host_bufferD[i * sample_size_sqrt + j]);
		}
		printf("\n");
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
