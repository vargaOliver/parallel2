#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

const int SAMPLE_SIZE = 100;

int* generateVector()
{
	int *toReturn = malloc(SAMPLE_SIZE * sizeof(int));
	for (int i = 0; i < SAMPLE_SIZE; ++i) {
		toReturn[i] = rand() % 100;
	}
	return toReturn;
}

int main(void)
{
    int i;
    cl_int err;
	int error_code;

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
        return 0;
    }
    cl_kernel kernel = clCreateKernel(program, "hello_kernel", NULL);

    // Create the host buffer and initialize it
    int* host_bufferA = (int*)malloc(SAMPLE_SIZE * sizeof(int));
	int* host_bufferB = (int*)malloc(SAMPLE_SIZE * sizeof(int));
	int* host_bufferC = (int*)malloc(SAMPLE_SIZE * sizeof(int));
	
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
    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

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

    for (i = 0; i < SAMPLE_SIZE; i+=1) {
        printf("A[%d] = %d, ", i, host_bufferA[i]);
		printf("B[%d] = %d, ", i, host_bufferB[i]);
		printf("C[%d] = %d ", i, host_bufferC[i]);
		printf("\n");
    }

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(host_bufferA);
	free(host_bufferB);
	free(host_bufferC);
}
