#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define N 1024 //wielko�� obliczanych wektor�w
#define imin(a, b) (a<b?a:b)
const int threadsPerBlock = 256; //ilo�� w�tk�w na k�zdy blok
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//ilo�� wykorzystywanych blok�w

__global__ void multiplyMatrix(float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock]; //Zmienna dzielona ze wszystkimi w�tkami w tym bloku. Nie dzieli si� z innymi blokami!

	int tid = threadIdx.x + blockIdx.x * blockDim.x; //id w�tku kt�ry to wykonuje, id w�tku + id bloku * pojemno�� bolku
	int cacheIndex = threadIdx.x; //id cache, kt�re jest takie samo jak id obecnego w�tku 

	float temp = 0;
	while (tid < N) {
		temp = a[tid] * b[tid]; //zapis mno�enia w zmiennej 
		tid += blockDim.x * gridDim.x; //przesuwanie o ilo�� wszystkich w�tk�w w ca�ej siatce, nie trzeba ogarnia� na czw�rk� 
	}

	cache[cacheIndex] = temp; //przypisanie wyniku mno�enia do wsp�dzielonej tablicy cache

	__syncthreads(); //czekanie a� wszystkie w�tki dotr� to tego miejsca

	//tu troch� w powalony spos�b sumuj� si� wszystkie wyniki
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}

		__syncthreads();
		i /= 2;
	}
	//przypisanie sumy wszystkich wynik�w mno�enia do tablicy c
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0]; //jako, �e cache nie jest wsp�dzielony pomi�dzy blokami to wynik�w b�dzie tyle ile by�o wykorzystanych blok�w, p�niej to si� sumuje na cpu

}




int main(void) {
	float *a, *b, c, *partial_c; //deklarowanie tablic cpu
	float *dev_a, *dev_b, *dev_partial_c; //a tu gpu device

	//umieszczanie zmiennych w pami�ci CPU
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid * sizeof(float));
	//umieszczanie zmiennych w pami�ci GPU
	cudaMalloc((void**)&dev_a, N * sizeof(float));
	cudaMalloc((void**)&dev_b, N * sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));
	//jakie� uzupe�nianie tablic z cpu
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}
	//kopiowanie zawarto�ci tablic z cpu do tablic na gpu
	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
	//wykonywanie funkcji na device, tu deklarujemy ilo�� wykorzystanych blok�w oraz w�tk�w na ka�dy blok (w obu zadaniach na 4 wystarczy�o <<<1, 1>>> jeden blok i jeden w�tek)
	multiplyMatrix << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);
	//kopiowanie wynik�w z gpu na cpu
	cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	//sumowanie wszystkich wynik�w z r�znych blok�w ju� musi si� wykona� na cpu
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		c += partial_c[i];
	}

	printf("Matrix A: ");
	for (int i = 0; i < N; i++)
		printf("%.0f ", a[i]);
	printf("\nMatrix B: ");
	for (int i = 0; i < N; i++)
		printf("%.0f ", b[i]);
	printf("\nA * B: %.0f", c);
	//zwalnianie pami�ci z gpu
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	//zwalnianie pami�ci z cpu
	free(a);
	free(b);
	free(partial_c);

	getchar();
	return 0;

}