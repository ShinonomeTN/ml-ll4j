package com.shinonometn.ml.ll4j.demo;

import com.shinonometn.tools.MNIST;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ConvertMNISTDatasetToCSV {
    /* MNIST Training Label Set */
    private final static String LABEL_PATH = "./digits/t10k-labels-idx1-ubyte";
    /* MNIST Training Image Data */
    private final static String IMAGE_PATH = "./digits/t10k-images-idx3-ubyte";
    /* CSV Output Path */
    private final static String OUTPUT_PATH = "./digits/test-images.csv";

    public static void main(String[] args) throws Exception {
        System.out.println("Label Path  : " + LABEL_PATH);
        System.out.println("Image Path  : " + IMAGE_PATH);
        System.out.println("Output Path : " + OUTPUT_PATH);

        System.out.println("Convert MNIST Data set to CSV...");

        // Load labels
        System.out.println("Open labels...");
        final MNIST.EntryIterator labelIterator = createDataSetIterator(Paths.get(LABEL_PATH));
        final int[] labels = new int[labelIterator.entryCount];
        int labelIndex = 0;
        while(labelIterator.hasNext()) {
            MNIST.Entry it = labelIterator.next();
            labels[labelIndex++] = it.data[0];
        }
        labelIterator.close();
        System.out.printf("Label Count: %d%n", labelIndex + 1);

        // Load images
        System.out.println("Open images...");
        final MNIST.EntryIterator imageIterator = createDataSetIterator(Paths.get(IMAGE_PATH));

        /* Write CSV header */
        final PrintWriter writer = new PrintWriter(OUTPUT_PATH);
        writer.print("label");
        for (int i = 0; i < imageIterator.getEntrySize(); i++) writer.print(",pixel" + i);
        writer.println();
        writer.flush();
        System.out.println("CSV header written.");

        /* Write images */
        int imageIndex = 0;
        while(imageIterator.hasNext()) {
            MNIST.Entry it = imageIterator.next();
            final int label = labels[imageIndex];
            final int w = it.getDimensionSize(0);
            final int h = it.getDimensionSize(1);
            final byte[] data = it.data;
            writer.print(label);
            for (int x = 0; x < w; x++) {
                for (int y = 0; y < h; y++) {
                    // normalize
                    final double value = (double) (data[x * h + y] & 0xFF) / 255;
                    writer.write("," + value);
                }
            }
            writer.println();
            imageIndex++;
        }
        if (imageIndex != labelIndex) System.out.println(">> Warning: label count != image count <<");
        writer.flush();
        writer.close();
        System.out.println("File saved. total " + (imageIndex + 1) + " images.");
    }

    //================================================================
    private static MNIST.EntryIterator createDataSetIterator(final Path path) throws IOException {
        final MNIST.EntryIterator iterator = MNIST.createDataSetIterator(path);
        System.out.printf(
                "Data type: 0x%02x %s, dimensions: %d%n",
                iterator.type.code,
                iterator.type.description,
                iterator.dataSetDimensions
        );
        for (int i = 0; i < iterator.getEntryDimensions(); i++) {
            System.out.printf("Entry Dimension %d size: %d%n", i + 1, iterator.getEntryDimensionSize(i));
        }
        return iterator;
    }
}
