package com.shinonometn.ml.ll4j.test;

import com.shinonometn.ml.ll4j.DataSet;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

public class DataSetLoadTest {
    private final static String FilePath = "./ll4j-demo/playground/digits/train-images.csv";

    public static void main(String[] args) throws IOException {
        final AtomicInteger counter = new AtomicInteger(0);
        final DataSet.SampleIterator<DataSet.LabelEntry> iter = DataSet.LabelEntry.createCSVIterator(FilePath, true);
        iter.forEachRemaining(e -> {
            counter.incrementAndGet();
        });
        System.out.println("Total " + counter.get() + " elements.");
        iter.close();
    }
}
