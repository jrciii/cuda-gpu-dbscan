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

        String file = CudaGdbscan.class.getResource("/getNeighbors8_0.ptx").getFile().replace("/C:","C:");
        CUmodule module = new CUmodule();
        cuModuleLoad(module, file);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function,module,"cudaGetNeighbors");

        int length = xs.length;

        Pointer hostPointsX = Pointer.to(xs);
        Pointer hostPointsY = Pointer.to(ys);

        CUdeviceptr cudaPointsX = new CUdeviceptr();
        cuMemAlloc(cudaPointsX,length * Sizeof.DOUBLE);
        cuMemcpyHtoD(cudaPointsX, hostPointsX, length * Sizeof.DOUBLE);

        CUdeviceptr cudaPointsY = new CUdeviceptr();
        cuMemAlloc(cudaPointsY,length * Sizeof.DOUBLE);
        cuMemcpyHtoD(cudaPointsY, hostPointsY, length * Sizeof.DOUBLE);

        CUdeviceptr cudaNeighborArray = new CUdeviceptr();
        cuMemAlloc(cudaNeighborArray,length * length * Sizeof.INT);
        cudaMemset(cudaNeighborArray, 0, length * length);

        CUdeviceptr cudaVis = new CUdeviceptr();
        cuMemAlloc(cudaVis,length * Sizeof.INT);
        cudaMemset(cudaVis, -1, length);
        Pointer kernelParameters = Pointer.to(Pointer.to(cudaPointsX), Pointer.to(cudaPointsY), Pointer.to(cudaVis), Pointer.to(new int[]{length}), Pointer.to(cudaNeighborArray), Pointer.to(new double[]{eps}), Pointer.to(new int[]{minPts}));

        cuLaunchKernel(function,10,1,1,10,1,1,0, null, kernelParameters, null);

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
