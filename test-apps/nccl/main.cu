/*%****************************************************************************80
  %  Code: 
  %   ncclSendRecv.cu
  %
  %  Purpose:
  %   Implements sample send/recv code using the package NCCL (p2p).
  %
  %  Modified:
  %   Aug 18 2020 10:57 
  %
  %  Author:
  %   Murilo Boratto <murilo.boratto 'at' fieb.org.br>
  %
  %  How to Compile:
  %   nvcc ncclSendRecv.cu -o object -lnccl  
  %
  %  HowtoExecute: 
  %   ./object 
  %                         
  %****************************************************************************80*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nccl.h>
#include <iostream>

#define CHECK_PEER_ACCESS(sourceDevice, destDevice) \
  do { \
    int canAccessPeer; \
    cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, sourceDevice, destDevice); \
    if (err != cudaSuccess) { \
      printf("cudaDeviceCanAccessPeer failed with error %s\n", cudaGetErrorString(err)); \
      exit(1); \
    } \
    if (!canAccessPeer) { \
      printf("Error: CUDA devices %d and %d cannot access each other's memory\n", sourceDevice, destDevice); \
      exit(1); \
    } \
  } while (0)


#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      printf("CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

__global__ void kernel(int *a, int rank) { 

  if(rank == 0)
    printf("%d\t", a[threadIdx.x]); 
  else
    printf("%d\t", a[threadIdx.x]*10); 
}

void show_all(int *in, int n){

  printf("\n");

  for(int i=0; i < n; i++)
    printf("%d\t", in[i]);

  printf("\n");

}/*show_all*/


int main(int argc, char* argv[]) {

  cudaSetDevice(0);
  cudaDeviceDisablePeerAccess(1);
  cudaDeviceDisablePeerAccess(2);
  cudaDeviceDisablePeerAccess(3);
  cudaSetDevice(1);
  cudaDeviceDisablePeerAccess(0);
  cudaDeviceDisablePeerAccess(2);
  cudaDeviceDisablePeerAccess(3);
  cudaSetDevice(2);
  cudaDeviceDisablePeerAccess(0);
  cudaDeviceDisablePeerAccess(1);
  cudaDeviceDisablePeerAccess(3);
  cudaSetDevice(3);
  cudaDeviceDisablePeerAccess(0);
  cudaDeviceDisablePeerAccess(1);
  cudaDeviceDisablePeerAccess(2);
  cudaSetDevice(0);

  // CHECK_PEER_ACCESS(0, 1);
  // CHECK_PEER_ACCESS(0, 2);
  // CHECK_PEER_ACCESS(0, 3);

  // CHECK_PEER_ACCESS(1, 0);
  // CHECK_PEER_ACCESS(1, 2);
  // CHECK_PEER_ACCESS(1, 3);

  // CHECK_PEER_ACCESS(2, 0);
  // CHECK_PEER_ACCESS(2, 1);
  // CHECK_PEER_ACCESS(2, 3);

  // CHECK_PEER_ACCESS(3, 0);
  // CHECK_PEER_ACCESS(3, 1);
  // CHECK_PEER_ACCESS(3, 2);

  int size = 8;

  /*Get current amounts number of GPU*/
  int nGPUs = 0;
  cudaGetDeviceCount(&nGPUs);

  
  std::cout << "Got Device Count" << std::endl;
  printf("nGPUs = %d\n",nGPUs);

  /*List GPU Device*/
  int *DeviceList = (int *) malloc ( nGPUs * sizeof(int));
  std::cout << "Allocated Device List" << std::endl;

  for(int i = 0; i < nGPUs; ++i)
    DeviceList[i] = i;

  /*NCCL Init*/
  ncclComm_t* comms         = (ncclComm_t*)  malloc(sizeof(ncclComm_t)  * nGPUs);  
  cudaStream_t* s           = (cudaStream_t*)malloc(sizeof(cudaStream_t)* nGPUs);
  ncclCommInitAll(comms, nGPUs, DeviceList); 
  std::cout << "Initiated NCCL Communication" << std::endl;

  /*General variables*/
  int *host       = (int*) malloc(size      * sizeof(int));
  int **sendbuff  = (int**)malloc(nGPUs     * sizeof(int*));
  int **recvbuff  = (int**)malloc(nGPUs     * sizeof(int*));

  /*Population of vector*/
  for(int i = 0; i < size; i++)
    host[i] = i + 1;

  show_all(host, size);

  for(int g = 0; g < nGPUs; g++) {
    cudaSetDevice(DeviceList[g]);
    cudaStreamCreate(&s[g]);
    std::cout << "Created Stream for device " << g << std::endl;
    cudaMalloc(&sendbuff[g], size * sizeof(int));
    cudaMalloc(&recvbuff[g], size * sizeof(int));

    if(g == 0)
      cudaMemcpy(sendbuff[g], host, size * sizeof(int),cudaMemcpyHostToDevice);

  }/*for*/

  ncclGroupStart();        
  std::cout << "Started Group" << std::endl;

  for(int g = 0; g < nGPUs; g++) {
    ncclSend(sendbuff[0], size, ncclInt, g, comms[g], s[g]);
    std::cout << "Sent Data" << std::endl;
    ncclRecv(recvbuff[g], size, ncclInt, g, comms[g], s[g]);
    std::cout << "Recv Data" << std::endl;
  }

  ncclGroupEnd();          
  std::cout << "Group Ended" << std::endl;

  for(int g = 0; g < nGPUs; g++) {
    cudaSetDevice(DeviceList[g]);
    printf("\nThis is device %d\n", g);
    if(g==0)
      kernel <<< 1 , size >>> (sendbuff[g], g); 
    else
      kernel <<< 1 , size >>> (recvbuff[g], g); 
    cudaDeviceSynchronize();
  }

  printf("\n");

  for (int g = 0; g < nGPUs; g++) {
    cudaSetDevice(DeviceList[g]);
    cudaStreamSynchronize(s[g]);
  }


  for(int g = 0; g < nGPUs; g++) {
    cudaSetDevice(DeviceList[g]);
    cudaStreamDestroy(s[g]);
  }

  for(int g = 0; g < nGPUs; g++) {
    ncclCommDestroy(comms[g]);
  }

  free(s);
  free(host);

  cudaFree(sendbuff);
  cudaFree(recvbuff);

  return 0;

}/*main*/

