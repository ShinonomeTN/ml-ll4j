package huzpsb.ll4j.samples;

import huzpsb.ll4j.minrt.MinRt;

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

public class TestMinRt {

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
        public final double[] data;

        LabeledData(int label, double[] data) {
            this.label = label;
            this.data = data;
        }
    }

    static void dumpAsImage(LabeledData data) throws IOException {
        final double[] imageData = data.data;
        final BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 28 * 28; i++) {
            int x = i % 28;
            int y = i / 28;
            int rgb = (int) (imageData[i] * 255);
            img.setRGB(x, y, rgb << 16 | rgb << 8 | rgb);
        }
        ImageIO.write(img, "png", new File("test.png"));
    }

    static Iterator<LabeledData> createDataIterator(String dataSetPath) throws IOException {
        final Scanner scanner = new Scanner(Files.newInputStream(Paths.get(dataSetPath)));
        scanner.nextLine(); // Skip the csv header

        return new Iterator<LabeledData>() {
            @Override
            public boolean hasNext() {
                return scanner.hasNextLine();
            }

            @Override
            public LabeledData next() {
                final String[] line = scanner.nextLine().split(",");
                final double[] buffer = new double[line.length - 1];
                for (int i = 1; i < line.length; i++) buffer[i] = Double.parseDouble(line[i]);
                final int label = Integer.parseInt(line[0]);
                return new LabeledData(label, buffer);
            }
        };
    }

    final static String LabeledDataPath = "fashion-mnist_test.csv";

    public static void main(String[] args) throws Exception {

        final String[] model = loadModelString();
        final Iterator<LabeledData> sampleDataSet = createDataIterator(LabeledDataPath);

        // Catten Linger:
        // Here it loads a csv file.
        // It seems like a csv containing correct answers
        final Scanner sc = new Scanner(new File("fashion-mnist_test.csv"));
        sc.nextLine(); // Skip header

        int correct = 0, wrong = 0;
        // Catten Linger:
        // We can see here the program runs the test 100 times.
        for (int c = 0; c < 100; c++) {
            // Catten Linger:
            // Each line is a csv, rows are splited with ','
            String[] line = sc.nextLine().split(",");

            // Catten Linger:
            // Here it created an array 1 entity smaller than the input
            // It maybe means that the first field is a label
            double[] input = new double[line.length - 1];
            for (int i = 0; i < line.length - 1; i++) {
                input[i] = Double.parseDouble(line[i + 1]);
            }

            // Catten Linger:
            // Why this test?
            // It means "dump the first image at the start"...
            if (correct == 0 && wrong == 0) {
                BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
                for (int i = 0; i < 28 * 28; i++) {
                    int x = i % 28;
                    int y = i / 28;
                    int rgb = (int) (input[i] * 255);
                    img.setRGB(x, y, rgb << 16 | rgb << 8 | rgb);
                }
                ImageIO.write(img, "png", new File("test.png"));
            }

            int actualLabel = Integer.parseInt(line[0]);
            // Catten Linger:
            // Here we see, it put "script" and "input" into the "doAi" method.
            // I can smell some weird bad joke...
            int predictedLabel = MinRt.doAi(input, model);
            if (actualLabel == predictedLabel) {
                correct++;
            } else {
                wrong++;
            }
        }
        System.out.println("Correct: " + correct + " Wrong: " + wrong);
    }
}
