package com.jrciii;

import jcuda.*;
import jcuda.runtime.*;
import jcuda.driver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.driver.JCudaDriver.*;

import java.util.ArrayDeque;
import java.util.Arrays;

public class CudaGdbscan {
    public static int[] gdbscan(double[] xs, double[] ys, double eps, int minPts) {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, CudaGdbscan.class.getResource("/getNeighbors8_0.ptx").getFile());

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function,module,"cudaGetNeighbors");

        int length = xs.length;

        Pointer hostPointsX = Pointer.to(xs);
        Pointer hostPointsY = Pointer.to(ys);

        Pointer cudaPointsX = new Pointer();
        cudaMalloc(cudaPointsX,length * Sizeof.DOUBLE);

        Pointer cudaPointsY = new Pointer();
        cudaMalloc(cudaPointsY,length * Sizeof.DOUBLE);

        Pointer cudaNeighborArray = new Pointer();
        cudaMalloc(cudaNeighborArray,length * length * Sizeof.INT);
        cudaMemset(cudaNeighborArray, 0, length * length);

        Pointer cudaVis = new Pointer();
        cudaMalloc(cudaVis,length * Sizeof.INT);
        cudaMemset(cudaVis, -1, length);

        cudaMemcpy(cudaPointsX, hostPointsX, length, cudaMemcpyKind.cudaMemcpyHostToDevice);
        cudaMemcpy(cudaPointsY, hostPointsY, length, cudaMemcpyKind.cudaMemcpyHostToDevice);

        Pointer kernelParameters = Pointer.to(cudaPointsX, cudaPointsY, cudaVis, Pointer.to(new int[]{length}), cudaNeighborArray, Pointer.to(new double[]{eps}), Pointer.to(new int[]{minPts}));

        cuLaunchKernel(function,16,1,1,32,1,1,0, null, kernelParameters, null);

        cuCtxSynchronize();

        int[] hostNeighborArry = new int[length * length];
        int[] hostVis = new int[length];
        cudaMemcpy(Pointer.to(hostNeighborArry), cudaNeighborArray, length * length, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(hostVis), cudaVis, length, cudaMemcpyKind.cudaMemcpyDeviceToHost);

        int[] hostIds = new int[length];
        Arrays.fill(hostIds, -1);
        hostSetIds(hostIds, hostVis, length, hostNeighborArry);

        return hostIds;
    }

    static void hostSetIds(int[] ids, int[] vis, int len, int[] hostNeighbors) {
        ArrayDeque<Integer> s = new ArrayDeque<>(len);
        int t_idx = 1;
        for (int i = 0; i < len; i++) {
            if (vis[i] >= 0) {
                if (ids[i] < 1) {
                    ids[i] = t_idx;
                    int src = i * len;
                    for (int j = 0; j < len; j++) {
                        if (hostNeighbors[src + j] > 0) {
                            ids[j] = t_idx;
                            s.push(j);
                        }
                    }
                    while (!s.isEmpty()) {
                        if (vis[s.peek()] >= 0) {
                            src = s.peek() * len;
                            for (int j = 0; j < len; j++) {
                                if (hostNeighbors[src + j] > 0) {
                                    if (ids[j] < 1) {
                                        ids[j] = t_idx;
                                        s.push(j);
                                    }
                                }
                            }
                        }
                        s.pop();
                    }
                }
                ++t_idx;
            }
        }
    }
}
