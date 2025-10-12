package com.shinonometn.ml.ll4j.demo;

import com.shinonometn.ml.ll4j.DataSet;
import com.shinonometn.ml.ll4j.Model;
import com.shinonometn.utils.Loaders;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class FashionMnistClassify {

    // THe origin model
//    private final static String ModelPath = "../ll4j-huzpsb/src/main/resources/test.model";
    private final static String ModelPath = "test2.model";
    private final static String LabeledDataPath = "fashion-mnist_test.csv";

    static void dumpAsImage(double[] sampleData) throws IOException {
        final BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 28 * 28; i++) {
            int x = i % 28;
            int y = i / 28;
            int rgb = (int) (sampleData[i] * 255);
            img.setRGB(x, y, rgb << 16 | rgb << 8 | rgb);
        }
        ImageIO.write(img, "png", new File("test.png"));
    }

    public static void main(String[] args) throws Exception {
        System.out.printf("Model path : %s\n", Paths.get(ModelPath).toAbsolutePath());
        System.out.printf("Label path : %s\n", Paths.get(LabeledDataPath).toAbsolutePath());

        final Model model = Model.parseLayers(Loaders.loadModelString(ModelPath));

        final DataSet.SampleIterator<DataSet.LabelEntry> sampleDataSet = DataSet
                .LabelEntry.createCSVIterator(LabeledDataPath, true);

        int correct = 0, wrong = 0;
        System.out.println("Start testing...");
        final long startTime = System.currentTimeMillis();
        int count = 0;
        while (sampleDataSet.hasNext()) {
            final DataSet.LabelEntry data = sampleDataSet.next();
            if (count == 0) dumpAsImage(data.values);

            final int predictedLabel = (int) model.classification(data.values)[0];

            final int actualLabel = data.label;
            final boolean isCorrect = (predictedLabel == actualLabel);
            if (isCorrect) correct++;
            else wrong++;

            count++;
            if (count % 100 == 0) System.out.printf(
                    "\rItem: %d, label: %d, predicted: %d, correct: %s      ",
                    count, actualLabel, predictedLabel, isCorrect
            );
            if (count >= 10_000) break;
        }
        sampleDataSet.close();
        final long timeDiff = System.currentTimeMillis() - startTime;
        System.out.println();
        System.out.printf("Test sample count: %d, Correct: %d Wrong: %d%n", count, correct, wrong);
        System.out.printf("Correct Rate: %02.2f%n", (double) correct / (double) count * 100);

        System.out.printf("Time used: %f seconds.%n", timeDiff / 1000.0);
        System.out.printf("Average: %f ms/i.%n", timeDiff / (double) count);
    }
}
