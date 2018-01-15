package com.jrciii;

import jcuda.runtime.JCuda;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CudaGdbscanTest {
    @Test
    public void test1() throws IOException {
        try(BufferedReader brt = new BufferedReader(new FileReader("src/test/resources/test1.txt"));
            //BufferedReader bra = new BufferedReader(new FileReader(CudaGdbscanTest.class.getResource("answers1.txt").getFile()))
            ) {
            String line = null;
            List<Double> xs = new ArrayList<>();
            List<Double> ys = new ArrayList<>();
            while ((line = brt.readLine()) != null) {
                String[] s = line.split(" ");
                xs.add(Double.valueOf(s[0]));
                ys.add(Double.valueOf(s[1]));
            }
            double[] xsA = new double[xs.size()];
            double[] ysA = new double[ys.size()];

            for (int i = 0; i < xs.size(); ++i) {
                xsA[i] = xs.get(i);
            }

            for (int i = 0; i < ys.size(); ++i) {
                ysA[i] = ys.get(i);
            }
            JCuda.setExceptionsEnabled(true);
            int[] ids = CudaGdbscan.gdbscan(xsA,ysA,4.0,6);
            System.out.println(Arrays.toString(ids));
        }
    }
}
