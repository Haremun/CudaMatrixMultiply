#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define N 1024 //wielkoœæ obliczanych wektorów
#define imin(a, b) (a<b?a:b)
const int threadsPerBlock = 256; //iloœæ w¹tków na k¹zdy blok
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//iloœæ wykorzystywanych bloków

__global__ void multiplyMatrix(float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock]; //Zmienna dzielona ze wszystkimi w¹tkami w tym bloku. Nie dzieli siê z innymi blokami!

	int tid = threadIdx.x + blockIdx.x * blockDim.x; //id w¹tku który to wykonuje, id w¹tku + id bloku * pojemnoœæ bolku
	int cacheIndex = threadIdx.x; //id cache, które jest takie samo jak id obecnego w¹tku 

	float temp = 0;
	while (tid < N) {
		temp = a[tid] * b[tid]; //zapis mno¿enia w zmiennej 
		tid += blockDim.x * gridDim.x; //przesuwanie o iloœæ wszystkich w¹tków w ca³ej siatce, nie trzeba ogarniaæ na czwórkê 
	}

	cache[cacheIndex] = temp; //przypisanie wyniku mno¿enia do wspó³dzielonej tablicy cache

	__syncthreads(); //czekanie a¿ wszystkie w¹tki dotr¹ to tego miejsca

	//tu trochê w powalony sposób sumuj¹ siê wszystkie wyniki
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}

		__syncthreads();
		i /= 2;
	}
	//przypisanie sumy wszystkich wyników mno¿enia do tablicy c
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0]; //jako, ¿e cache nie jest wspó³dzielony pomiêdzy blokami to wyników bêdzie tyle ile by³o wykorzystanych bloków, póŸniej to siê sumuje na cpu

}




int main(void) {
	float *a, *b, c, *partial_c; //deklarowanie tablic cpu
	float *dev_a, *dev_b, *dev_partial_c; //a tu gpu device

	//umieszczanie zmiennych w pamiêci CPU
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid * sizeof(float));
	//umieszczanie zmiennych w pamiêci GPU
	cudaMalloc((void**)&dev_a, N * sizeof(float));
	cudaMalloc((void**)&dev_b, N * sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));
	//jakieœ uzupe³nianie tablic z cpu
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}
	//kopiowanie zawartoœci tablic z cpu do tablic na gpu
	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
	//wykonywanie funkcji na device, tu deklarujemy iloœæ wykorzystanych bloków oraz w¹tków na ka¿dy blok (w obu zadaniach na 4 wystarczy³o <<<1, 1>>> jeden blok i jeden w¹tek)
	multiplyMatrix << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);
	//kopiowanie wyników z gpu na cpu
	cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	//sumowanie wszystkich wyników z róznych bloków ju¿ musi siê wykonaæ na cpu
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
	//zwalnianie pamiêci z gpu
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	//zwalnianie pamiêci z cpu
	free(a);
	free(b);
	free(partial_c);

	getchar();
	return 0;

}