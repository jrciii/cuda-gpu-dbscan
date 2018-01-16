package com.jrciii;

import jcuda.runtime.JCuda;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.junit.Test;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class CudaGdbscanTest {
    @Test
    public void test1() throws IOException, InterruptedException {
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
            Map<Integer, XYSeries> clusters = new HashMap<>();
            for (int i = 0; i < ids.length; ++i) {
                XYSeries points = clusters.get(ids[i]);
                if (points == null) {
                    points = new XYSeries(ids[i]);
                    clusters.put(ids[i],points);
                }
                points.add(xsA[i],ysA[i]);
            }

            XYSeriesCollection ds = new XYSeriesCollection();
            clusters.values().forEach(v -> ds.addSeries(v));
            JFreeChart chart = ChartFactory.createScatterPlot("XY", "X", "Y", ds);
            ChartFrame frame = new ChartFrame("Clusters", chart);
            frame.pack();
            frame.setVisible(true);
            while(true) {
                Thread.sleep(5000);
            }
        }
    }
}
