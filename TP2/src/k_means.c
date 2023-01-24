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
float distanceCalculation(float clusterX, float clusterY, float pointX, float pointY){
    return (float) ((clusterX - pointX)*(clusterX - pointX)) + ((clusterY - pointY)*(clusterY - pointY));
}

void populate(int* numPoints, float *pointX, float *pointY,int *pointCluster, float *clusterX, float *clusterY, int n, int k, int t){
    srand(10);
        int i;
        for(int i = 0; i < n; i++){
            pointX[i] = (float) rand() / RAND_MAX;  // Atribuimos a cada ponto cordenadas random
            pointY[i] = (float) rand() / RAND_MAX;  // Atribuimos a cada ponto cordenadas random
            pointCluster[i] = -1;                   // Inicialmente cada amostra ainda não está associada a nenhum cluster
        }

        for(int i = 0; i < k; i++){
            clusterX[i] = pointX[i];                // Seguindo o algoritmo, os primeiros K pontos iniciais são os centroids dos K clusters.
            clusterY[i] = pointY[i];
            numPoints[i]++;                         // Atualizamos o numero de pontos associado a cada cluster
            pointCluster[i] = i;                    // Atribuimos ao ponto o cluster a que está associado
    }
}

// Função para calcular o melhor cluster para cada ponto.
int bestCluster(float* clusterX, float* clusterY, float pointX, float pointY,  int n, int k, int t){

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
    return bestCluster;
}




void calculateBestCluster(int* numPoints, float *pointX, float *pointY, int *pointCluster, float *clusterX, float *clusterY, int n, int k, int t){
    int bClu = -1;
    int i;
    #pragma omp parallel for num_threads(t) private(i, bClu) reduction(+:numPoints[:k])
    for(i = 0; i < n; i++){
        bClu = bestCluster(clusterX, clusterY, pointX[i], pointY[i], n, k, t);   // Calcula o melhor cluster para o ponto
        if(bClu != pointCluster[i]){    // Caso o melhor cluster seja diferente do atribuido anteriormente é atualizado
            if(pointCluster[i] != -1)
                numPoints[pointCluster[i]]--;   // Decrementamos o numero de pontos associado a esse cluster

            numPoints[bClu]++;  // Incrementamos o numero de pontos associado a esse cluster
        }
        pointCluster[i] = bClu;
    }
}

void calculateCentroid(float *pointX, float *pointY, int *pointCluster, float *clusterX, float *clusterY, int *numPoints,  int n, int k, int t){

    float xSumCluster[k];
    float ySumCluster[k];
    
    memset(xSumCluster, 0.0f, k*sizeof(float));
    memset(ySumCluster, 0.0f, k*sizeof(float));
   
    #pragma omp parallel num_threads(t) reduction(+:xSumCluster[:k]) reduction(+:ySumCluster[:k])
    {
        #pragma omp for //reduction(+:xSumCluster[:k] ySumCluster[:k])
        for(int i = 0; i < n; i++){
            xSumCluster[pointCluster[i]]  += pointX[i];
            ySumCluster[pointCluster[i]]  += pointY[i];
        }
    }
    
    for(int i = 0; i < k; i++){
        clusterX[i] = (float) xSumCluster[i] / numPoints[i];
        clusterY[i] = (float) ySumCluster[i] / numPoints[i];
    }
}


// Função de Print
void printClusters(float* clusterX, float* clusterY, int* numPoints, int n, int k, int t){
    printf("N = %d, K = %d\n", n, k);
    for(int i = 0; i < k; i++){
        printf("Center: (%.3f, %.3f) : Size: %d\n", clusterX[i], clusterY[i], numPoints[i]);
    }
}

// Função principal (Algoritmo)
void lloyds(float *pointX, float *pointY, int *pointCluster, float *clusterX, float *clusterY, int n, int k, int t){
  
    int *numPoints = malloc(k * sizeof(int));
    for(int i = 0; i < k; i++)
        numPoints[i] = 0;
    populate(numPoints, pointX, pointY, pointCluster, clusterX, clusterY,n, k , t);
    calculateBestCluster(numPoints, pointX, pointY, pointCluster, clusterX, clusterY, n, k, t);
    calculateCentroid(pointX, pointY, pointCluster, clusterX, clusterY, numPoints, n, k, t);
    
    int a = 0;
    while(a < 20){
        calculateBestCluster(numPoints, pointX, pointY, pointCluster, clusterX, clusterY, n, k, t);
        calculateCentroid(pointX, pointY, pointCluster, clusterX, clusterY, numPoints, n,k,t);
        a++;
    }

    printClusters(clusterX, clusterY, numPoints, n, k, t);
    printf("Iterations: %d\n", a);
    free(numPoints);
}

int main(int argc, char *argv[]){
    
    int n;
    int k;
    int t;
    if (argc  == 4) {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        t = atoi(argv[3]);
    } else { 
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        t = 1;
    } 

    float *pointX = malloc(n*sizeof(float));
    float *pointY = malloc(n*sizeof(float));

    int *pointCluster = malloc(n*sizeof(float));
    float *clusterX = malloc(k*sizeof(float));
    float *clusterY = malloc(k*sizeof(float));
    lloyds(pointX, pointY, pointCluster, clusterX, clusterY, n, k, t);
    free(pointX);
    free(pointY);
    free(pointCluster);
    free(clusterX);
    free(clusterY);
}

