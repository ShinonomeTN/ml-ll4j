package com.shinonometn.ml.ll4j.demo;

import com.shinonometn.ml.ll4j.DataSet;
import com.shinonometn.ml.ll4j.Model;
import com.shinonometn.utils.Loaders;
import com.shinonometn.utils.SampleVisualizingParams;
import com.shinonometn.utils.Visualizers;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MnistClassify {

    // The origin FashionMNIST model
    // private final static String ModelPath = "../ll4j-huzpsb/src/main/resources/test.model";

    private final static String ModelPath = Optional
            .ofNullable(System.getenv("MODEL_LOCATION"))
            .orElse("test2.model");

    private final static String LabeledDataPath = Optional
            .ofNullable(System.getenv("TEST_DATA_PATH"))
            .orElse("fashion-mnist_test.csv");

    // Background executor
    private static final ExecutorService executor = Executors.newSingleThreadExecutor();
    private static final Thread.UncaughtExceptionHandler uceHandler = (t, e) -> executor.execute(() -> {
        e.printStackTrace(System.err);
        executor.shutdown();
    });

    public static void main(String[] args) throws Exception {
        Thread.currentThread().setUncaughtExceptionHandler(uceHandler);

        System.out.printf("Model path : %s\n", Paths.get(ModelPath).toAbsolutePath());
        System.out.printf("Label path : %s\n", Paths.get(LabeledDataPath).toAbsolutePath());

        final Path wrongOutputPath = Paths.get("wrong_answers");
        if (wrongOutputPath.toFile().mkdirs()) System.out.println(
                "Created wrong output directory: " + wrongOutputPath.toAbsolutePath()
        );

        final Model model = Model.parseLayers(Loaders.loadModelString(ModelPath));

        final DataSet.SampleIterator<DataSet.LabelEntry> sampleDataSet = DataSet
                .LabelEntry.createCSVIterator(LabeledDataPath);

        int correct = 0, wrong = 0;
        System.out.println("Start testing...");
        final long startTime = System.currentTimeMillis();
        int count = 0;
        while (sampleDataSet.hasNext()) {
            final DataSet.LabelEntry data = sampleDataSet.next();

            final int predictedLabel = (int) model.classification(data.values)[0];

            final int actualLabel = data.getLabelValue();
            final boolean isCorrect = (predictedLabel == actualLabel);
            if (isCorrect) correct++; else wrong++;
            if (!isCorrect) executor.execute(() -> {
                try {
                    Visualizers.dumpSampleToGreyScaleImageFile(
                            SampleVisualizingParams.rowFirst(28, 28, data.values),
                            wrongOutputPath.resolve(String.format("P%d_A%d.png", predictedLabel, actualLabel))
                    );
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });

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

        executor.shutdown();
    }
}
