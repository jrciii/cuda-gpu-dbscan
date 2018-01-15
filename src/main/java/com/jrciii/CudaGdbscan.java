package com.jrciii;

import jcuda.*;
import jcuda.runtime.*;
import jcuda.driver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.driver.JCudaDriver.*;

import java.util.List;

public class CudaGdbscan {
    public List<Integer> gdbscan(double[] xs, double[] ys, double eps, int minPts) {
        int length = xs.length;

        Pointer hostPointsX = Pointer.to(xs);
        Pointer hostPointsY = Pointer.to(ys);

        Pointer cudaPointsX = new Pointer();
        cudaMalloc(cudaPointsX,length * Sizeof.DOUBLE);

        Pointer cudaPointsY = new Pointer();
        cudaMalloc(cudaPointsY,length * Sizeof.DOUBLE);

        Pointer cudaNeighborArray = new Pointer();
        cudaMalloc(cudaNeighborArray,length * length * Sizeof.INT);

        Pointer cudaVis = new Pointer();
        cudaMalloc(cudaVis,length * Sizeof.INT);

        cudaMemcpy(cudaPointsX, hostPointsX, length, cudaMemcpyKind.cudaMemcpyHostToDevice);
        cudaMemcpy(cudaPointsY, hostPointsY, length, cudaMemcpyKind.cudaMemcpyHostToDevice);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, CudaGdbscan.class.getResource("getNeighbors.ptx").getFile());

        Pointer kernelParameters = Pointer.to(cudaPointsX, cudaPointsY, cudaVis, Pointer.to(new int[]{length}), cudaNeighborArray, Pointer.to(new double[]{eps}), Pointer.to(new int[]{minPts}));
    }
}
