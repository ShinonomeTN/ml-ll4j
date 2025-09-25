package com.shinonometn.ml.ll4j.test;

import com.shinonometn.ml.ll4j.MinRt;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

public class MinRtTest {

    final static String ModelPath = "test.model";

    static String[] loadModelString() throws IOException {
        List<String> buffer = new ArrayList<>();

        try (final Scanner scanner = new Scanner(Files.newInputStream(Paths.get(ModelPath)))) {
            while (scanner.hasNextLine()) buffer.add(scanner.nextLine());
        }

        return buffer.toArray(new String[0]);
    }

    static class LabeledData {
        public final int label;
        public final double[] payload;

        LabeledData(int label, double[] payload) {
            this.label = label;
            this.payload = payload;
        }
    }

    static void dumpAsImage(LabeledData data) throws IOException {
        final double[] imageData = data.payload;
        final BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 28 * 28; i++) {
            int x = i % 28;
            int y = i / 28;
            int rgb = (int) (imageData[i] * 255);
            img.setRGB(x, y, rgb << 16 | rgb << 8 | rgb);
        }
        ImageIO.write(img, "png", new File("test.png"));
    }

    static Iterator<LabeledData> createDataIterator(String dataSetPath, boolean skipHeader) throws IOException {
        final Scanner scanner = new Scanner(Files.newInputStream(Paths.get(dataSetPath)));
        if (skipHeader) {
            final String header = scanner.nextLine().trim(); // Skip the csv header
            System.out.println("Header: " + header);
        }

        return new Iterator<LabeledData>() {
            @Override
            public boolean hasNext() {
                return scanner.hasNextLine();
            }

            @Override
            public LabeledData next() {
                final String[] line = scanner.nextLine().split(",");
                final double[] buffer = new double[line.length - 1];
                for (int i = 1; i < line.length; i++) buffer[i - 1] = Double.parseDouble(line[i]);
                final int label = Integer.parseInt(line[0]);
                return new LabeledData(label, buffer);
            }
        };
    }

    final static String LabeledDataPath = "fashion-mnist_test.csv";

    public static void main(String[] args) throws Exception {

        final MinRt.Model parseModel = MinRt.parseModel(loadModelString());

        final Iterator<LabeledData> sampleDataSet = createDataIterator(LabeledDataPath, true);
        int correct = 0, wrong = 0;
        System.out.println("Start testing...");
        final long startTime = System.currentTimeMillis();
        int count = 0;
        while (sampleDataSet.hasNext()) {
            final LabeledData data = sampleDataSet.next();
            if (count == 0) dumpAsImage(data);

            final int predictedLabel = (int) MinRt.classification(data.payload, parseModel)[0];

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
        final long timeDiff = System.currentTimeMillis() - startTime;
        System.out.println();
        System.out.printf("Test sample count: %d, Correct: %d Wrong: %d%n", count, correct, wrong);
        System.out.printf("Correct Rate: %.2f%n", (double) correct / (double) count * 100);

        System.out.printf("Time used: %f seconds.%n", timeDiff / 1000.0);
        System.out.printf("Average: %f ms/i.%n", timeDiff / (double) count);
    }
}
