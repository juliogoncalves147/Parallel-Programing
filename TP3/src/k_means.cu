#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <string.h>

#define TRUE 1
#define FALSE 0

// Função que calcula a distância entre um ponto e um cluster
__device__ float distanceCalculation(float clusterX, float clusterY, float pointX, float pointY)
{
    return (float)((clusterX - pointX) * (clusterX - pointX)) + ((clusterY - pointY) * (clusterY - pointY));
}

void populate(int *numPoints, float *pointX, float *pointY, int *pointCluster, float *clusterX, float *clusterY, int n, int k, int t)
{
    srand(10);

    for (int i = 0; i < n; i++)
    {
        pointX[i] = (float)rand() / RAND_MAX; // Atribuimos a cada ponto cordenadas random
        pointY[i] = (float)rand() / RAND_MAX; // Atribuimos a cada ponto cordenadas random
        pointCluster[i] = -1;                 // Inicialmente cada amostra ainda não está associada a nenhum cluster
    }

    for (int i = 0; i < k; i++)
    {
        clusterX[i] = pointX[i]; // Seguindo o algoritmo, os primeiros K pontos iniciais são os centroids dos K clusters.
        clusterY[i] = pointY[i];
        numPoints[i]++;      // Atualizamos o numero de pontos associado a cada cluster
        pointCluster[i] = i; // Atribuimos ao ponto o cluster a que está associado
    }
}

// Função para calcular o melhor cluster para cada ponto.
/*
__device__
void bestCluster(float* clusterX, float* clusterY, float pointX, float pointY,  int n, int k, int t){

    int bestCluster = 0;
    float tempDistance = 0;
    float distance = distanceCalculation(clusterX[0], clusterY[0], pointX, pointY);
    for(int i = 1; i < k; i++){
        tempDistance = distanceCalculation(clusterX[i], clusterY[i], pointX, pointY);
        if(tempDistance < distance){
            distance = tempDistance;
            bestCluster = i;
        }
    }
}
*/

__global__ void calculateBestCluster(int *numPoints, float *pointX, float *pointY, int *pointCluster, float *clusterX, float *clusterY, int n, int k, int t)
{

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
    {
        int bClu = -1;
        // bClu = bestCluster(clusterX, clusterY, pointX[i], pointY[i], n, k, t);   // Calcula o melhor cluster para o ponto
        int bestCluster = 0;
        float tempDistance = 0;
        float distance = distanceCalculation(clusterX[0], clusterY[0], pointX[i], pointY[i]);
        for (int j = 1; j < k; j++)
        {
            tempDistance = distanceCalculation(clusterX[j], clusterY[j], pointX[i], pointY[i]);
            if (tempDistance < distance)
            {
                distance = tempDistance;
                bestCluster = i;
            }
        }

        bClu = bestCluster;

        if (bClu != pointCluster[i])
        { // Caso o melhor cluster seja diferente do atribuido anteriormente é atualizado
            if (pointCluster[i] != -1)
                numPoints[pointCluster[i]]--; // Decrementamos o numero de pontos associado a esse cluster

            numPoints[bClu]++; // Incrementamos o numero de pontos associado a esse cluster
        }
        pointCluster[i] = bClu;
    }

    __syncthreads();
}

void calculateCentroid(float *pointX, float *pointY, int *pointCluster, float *clusterX, float *clusterY, int *numPoints, int n, int k, int t)
{

    float xSumCluster[k];
    float ySumCluster[k];

    memset(xSumCluster, 0.0f, k * sizeof(float));
    memset(ySumCluster, 0.0f, k * sizeof(float));

#pragma omp parallel num_threads(t) reduction(+                              \
                                              : xSumCluster[:k]) reduction(+ \
                                                                           : ySumCluster[:k])
    {
#pragma omp for // reduction(+:xSumCluster[:k] ySumCluster[:k])
        for (int i = 0; i < n; i++)
        {
            xSumCluster[pointCluster[i]] += pointX[i];
            ySumCluster[pointCluster[i]] += pointY[i];
        }
    }

    for (int i = 0; i < k; i++)
    {
        clusterX[i] = (float)xSumCluster[i] / numPoints[i];
        clusterY[i] = (float)ySumCluster[i] / numPoints[i];
    }
}

// Função de Print
void printClusters(float *clusterX, float *clusterY, int *numPoints, int n, int k, int t)
{
    printf("N = %d, K = %d\n", n, k);
    for (int i = 0; i < k; i++)
    {
        printf("Center: (%.3f, %.3f) : Size: %d\n", clusterX[i], clusterY[i], numPoints[i]);
    }
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        printf("Cuda error: %s, %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

// Função principal (Algoritmo)
void lloyds(float *pointX, float *pointY, int *pointCluster, float *clusterX, float *clusterY, int n, int k, int t)
{

    int *numPoints = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++)
        numPoints[i] = 0;

    populate(numPoints, pointX, pointY, pointCluster, clusterX, clusterY, n, k, t);

    int *d_numPoints;
    float *d_pointX;
    float *d_pointY;
    int *d_pointCluster;
    float *d_clusterX;
    float *d_clusterY;
    //    int * d_n;
    //    int * d_k;
    //    int * d_t;

    cudaMalloc((void **)&d_numPoints, k * sizeof(int));
    cudaMalloc((void **)&d_pointX, n * sizeof(float));
    cudaMalloc((void **)&d_pointY, n * sizeof(float));
    cudaMalloc((void **)&d_pointCluster, n * sizeof(int));
    cudaMalloc((void **)&d_clusterX, k * sizeof(float));
    cudaMalloc((void **)&d_clusterY, k * sizeof(float));
    //    cudaMalloc((void**)&d_n, sizeof(int));
    //    cudaMalloc((void**)&d_k, sizeof(int));
    //    cudaMalloc((void**)&d_t, sizeof(int));

    cudaMemcpy(d_numPoints, numPoints, k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointX, pointX, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointY, pointY, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointCluster, pointCluster, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusterX, clusterX, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusterY, clusterY, k * sizeof(float), cudaMemcpyHostToDevice);
    //  cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    //  cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    //  cudaMemcpy(d_t, &t, sizeof(int), cudaMemcpyHostToDevice);

    calculateBestCluster<<<65535, 1024>>>(d_numPoints, d_pointX, d_pointY, d_pointCluster, d_clusterX, d_clusterY, n, k, t);

    cudaMemcpy(numPoints, d_numPoints, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pointCluster, d_pointCluster, n * sizeof(int), cudaMemcpyDeviceToHost);

    calculateCentroid(pointX, pointY, pointCluster, clusterX, clusterY, numPoints, n, k, t);

    int a = 0;
    while (a < 20)
    {

        cudaMemcpy(d_clusterX, clusterX, k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_clusterY, clusterY, k * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("memcpy1");

        calculateBestCluster<<<65535, 1024>>>(d_numPoints, d_pointX, d_pointY, d_pointCluster, d_clusterX, d_clusterY, n, k, t);

        cudaMemcpy(numPoints, d_numPoints, k * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pointCluster, d_pointCluster, n * sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError("memcpy2");

        calculateCentroid(pointX, pointY, pointCluster, clusterX, clusterY, numPoints, n, k, t);

        a++;
    }

    //   free(d_numPoints);
    //   free(d_pointX);
    //   free(d_pointY);
    //   free(d_pointCluster);
    //   free(d_clusterX);
    //   free(d_clusterY);
    //    free(d_n);
    //    free(d_k);
    //    free(d_t);

    printClusters(clusterX, clusterY, numPoints, n, k, t);
    printf("Iterations: %d\n", a);
    free(numPoints);
}

int main(int argc, char *argv[])
{

    int n;
    int k;
    int t;
    if (argc == 4)
    {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        t = atoi(argv[3]);
    }
    else
    {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        t = 1;
    }

    float *pointX = (float *)malloc(n * sizeof(float));
    float *pointY = (float *)malloc(n * sizeof(float));

    int *pointCluster = (int *)malloc(n * sizeof(float));

    float *clusterX = (float *)malloc(k * sizeof(float));
    float *clusterY = (float *)malloc(k * sizeof(float));

    lloyds(pointX, pointY, pointCluster, clusterX, clusterY, n, k, t);
    // tratar de passar a alocação de memoria para o cuda, transferencia de valores.
    free(pointX);
    free(pointY);
    free(pointCluster);
    free(clusterX);
    free(clusterY);
}
