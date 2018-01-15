package com.jrciii;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import org.junit.Test;

public class MinimalTest {
    @Test
    public void minimalTest() {
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: "+pointer);
        JCuda.cudaFree(pointer);
    }
}
