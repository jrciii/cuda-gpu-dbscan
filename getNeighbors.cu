extern "C"
__global__ void cudaGetNeighbors(double* xs, double* ys, int* vis, int len, int* neighbors, double minEps, int minPts) {

	unsigned int	tid	= blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int	src;
	unsigned int	dest;
	unsigned int	point_id = tid;
	unsigned int	neighborscnt;

	while (point_id < len * len) {
		src = point_id / len;
		dest = point_id % len;
		double dist = 0.0;
		if (src <= dest) {
			double srcX = xs[src];
			double destX = xs[dest];
			double srcY = ys[src];
			double destY = ys[dest];
			double xRes = srcX - destX;
			double yRes = srcY - destY;
			dist = xRes * xRes + yRes * yRes;
			if (dist < minEps * minEps) {
				neighbors[point_id] = 1;
			}
			neighbors[dest * len + src] = neighbors[point_id];
		}
		point_id += blockDim.x * gridDim.x;
	}

	__syncthreads();

	point_id = tid;
	while (point_id < len) {
		neighborscnt = 0;
		src = point_id * len;
		for (int i = 0; i < len; i++) {
			if (point_id != i) {
				if (neighbors[src + i]) {
					neighborscnt++;
				}
			}
		}
		if (neighborscnt >= minPts) {
			vis[point_id]++;
		}
		point_id += blockDim.x * gridDim.x;
	}
}
