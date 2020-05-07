#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstring>
#include <ctime>
#define N 1024*4
// Device Kernel
__global__ void amean(float *A, float *S)
 {
   	//holds intermediates in shared memory reduction
    	__shared__ int sdata[N];

    	int tid=threadIdx.x;
    	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid]=A[i];
        __syncthreads();

  	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
		   sdata[tid] += sdata[tid + s];
		 }
	 	__syncthreads();
   	}	

   	if(tid==0) 
	   S[blockIdx.x]=sdata[0];

}

//host Function
float cpu_amean(float *A) {

	float S,am;
	
     for(int i = 0; i < N; i++) {
        	S= S+A[i];
   	}

  	am=S/float(N);	
	return am;
	
}  
          
// host code
int main()
{
   
	size_t size = N * sizeof(float);
	FILE *f;
	f=fopen("amean.txt","w");
	float S;
	clock_t start,stop;
	   
   	printf("\nName of the Model= Data Parallel Model\n");

	float* d_A;
	float* d_S;
        
    	
    int threadsPerBlock;
	if (N<=1024) 
		threadsPerBlock=N;
	else 
		threadsPerBlock=1024;
		
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
	printf("\nblocksPerGrid=%d\n",blocksPerGrid);
	

	float* h_A = (float*)malloc(size);				// Allocate input vectors h_A and h_B in host memory
	float* h_S = (float*)malloc(sizeof(float)*blocksPerGrid);
	srand(time(NULL));
	for(int i = 0; i < N; i++) {					// Initialize input vectors
        	h_A[i] = rand()%100;
    	}

	cudaMalloc(&d_A, size);						// Allocate vector in device memory
	cudaMalloc(&d_S, sizeof(float)*blocksPerGrid);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 		// Copy vectors from host memory to device memory
   	//printf("\nAfter HostToDevice Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));
 	
 	//Parallel Computation
 	start = std::clock();
    	amean<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_S);		// Invoke kernel
	stop = std::clock();		
	 
     long int time=stop - start;
   	//printf("\nAfter global call Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));
    
    cudaMemcpy(h_S, d_S, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);	// Copy result from device memory to host memory
   	//printf("\nAfter DeviceToHost Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));
      
	for(int i = 1; i < blocksPerGrid; i++) {
       		h_S[0]+= h_S[i];
    	}
	S=h_S[0]/float(N);
	
	//Storing in file	
      	for(int i=0;i<N;i++)
	{
		fprintf(f,"%f ",h_A[i]);              //if correctly computed, then all values must be N
		fprintf(f,"\n");
	}

	fprintf(f,"Cuda Result= %f ",S);
	printf("_________________________Parallel______________________________\n	\n");
	printf("Cuda Result= %f ",S);
	printf("\n\nExecution Time of parallel Implementation= %ld (ms)\n",time);
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_S);
	printf("\n_________________________Sequential____________________________________\n	");
	
  	//Sequential Computations
  	start = std::clock();
  	S=cpu_amean(h_A);						// Invoke CPU function
  	stop = std::clock();
  	
  	//Storing in file
  	fprintf(f,"\nCPU Result= %f ",S);
	printf("\nCPU Result= %f ",S);
 	long int Stime=stop - start;
  	printf("\n\nExecution Time of Sequential Implementation= %ld (ms)\n",Stime );
	printf("_______________________________________________________________________	");
 	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
 	int cores;
	if (prop.major==2) //Fermi
		cores=(prop.minor==1) ? prop.multiProcessorCount*48 : prop.multiProcessorCount*32;

   	printf("\n\nNo. of cores:%d\n",cores);
   	printf("\nTotal cost=Execution Time * Number ofprocessors used = %ld",time*cores);
 	float eff=float(Stime)/float(time);
   	printf("\n\nSpeedup=WCSA / WCPA  =  %f\n",eff);
   	printf("\nEfficiency=Speedup/NumberOfProcessors =  %f\n\n",eff/cores);
	printf("_______________________________________________________________________	");
	
	// Free host memory      
	free(h_S);  
	free(h_A);
    
 
}
