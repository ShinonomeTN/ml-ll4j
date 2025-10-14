package com.shinonometn.ml.ll4j.demo;

import com.shinonometn.ml.ll4j.DataSet.LabelEntry;
import com.shinonometn.ml.ll4j.DataSet.SampleIterator;
import com.shinonometn.ml.ll4j.MinRtException;
import com.shinonometn.ml.ll4j.Model;
import com.shinonometn.ml.ll4j.ModelTrainer;
import com.shinonometn.utils.Formats;
import com.shinonometn.utils.Loaders;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.shinonometn.ml.ll4j.AdjustFunctions.fillWithGaussianRandom;
import static com.shinonometn.ml.ll4j.Layers.*;

public class MnistTrain {

    private final static String ModelLocation = Optional
            .ofNullable(System.getenv("MODEL_LOCATION"))
            .orElse("test2.model");

    private final static String LabeledDataPath = Optional
            .ofNullable(System.getenv("TRAIN_DATA_PATH"))
            .orElse("fashion-mnist_train.csv");

    // 8e-7 for fashion, 8e-5 for digits
    private final static double InitialLearningRate = Optional
            .ofNullable(System.getenv("LEARNING_RATE"))
            .map(Double::parseDouble)
            .orElse(ModelTrainer.DefaultLearningRate);

    private static final ExecutorService executor = Executors.newSingleThreadExecutor();
    private static final Thread.UncaughtExceptionHandler uceHandler = (t, e) -> executor.execute(() -> {
        e.printStackTrace(System.err);
        executor.shutdown();
    });

    private static void printProgressLine(final int c, final int t, final int f) {
        System.out.printf("\r[% 6d] t:% 6d, f:% 6d, r:%2.2f%%", c, t, f, ((t / (double) c) * 100));
    }

    public static void main(String[] args) throws IOException, MinRtException {
        Thread.currentThread().setUncaughtExceptionHandler(uceHandler);
        final Path ModelPath = Paths.get(ModelLocation);

        System.out.printf("Training data file : %s\n", Paths.get(LabeledDataPath).toAbsolutePath());
        System.out.printf("Model output file  : %s\n", ModelPath.toAbsolutePath());

        final ModelTrainer trainer;
        if (ModelPath.toFile().exists()) {
            trainer = ModelTrainer.on(Model.parseLayers(Loaders.loadModelString(ModelLocation)));
            System.out.println("Load origin weights from file.");
        } else {
            trainer = ModelTrainer.create(
                    fillWithGaussianRandom(dense(784, 100)),
                    leakyRelu(100),
                    fillWithGaussianRandom(dense(100, 100)),
                    leakyRelu(100),
                    fillWithGaussianRandom(dense(100, 10)),
                    judge(10)
            );
            System.out.println("New model weights created.");
        }

        final long start = System.currentTimeMillis();

        double learningRate = InitialLearningRate;

        for (int i = 0; i < 128; i++) {
            final long roundStart = System.currentTimeMillis();
            int trainCount = 0, correctCount = 0, wrongCount = 0;
            System.out.printf("======== Training round % 3d/128 ========\n", i + 1);
            try (final SampleIterator<LabelEntry> sampleDataSet =
                         LabelEntry.createCSVIterator(LabeledDataPath)) {

                while (sampleDataSet.hasNext()) {
                    // Adjust for single sample
                    final double correct = trainer.adjust(sampleDataSet.next(), learningRate);
                    if (correct > 0) correctCount++;
                    else wrongCount++;

                    if (((trainCount++) % 1000) != 0) continue;

                    final int c = trainCount;
                    final int t = correctCount;
                    final int f = wrongCount;
                    executor.execute(() -> printProgressLine(c, t, f));
                }

            }
            final long roundEnd = System.currentTimeMillis();
            printProgressLine(trainCount, correctCount, wrongCount);
            System.out.println();

            // Adjust the learning rate according to correct rate
            final double correctRate = (double) correctCount / trainCount;
            if (correctRate > 0.95) {
                learningRate = InitialLearningRate * 0.01;
            } else if (correctRate > 0.9) {
                learningRate = InitialLearningRate * 0.1;
            }

            System.out.printf(
                    "Round %03d finished, time: %s%n",
                    i + 1, Formats.millisDuration(roundEnd - roundStart)
            );
            trainer.writeModelToFile(ModelLocation);
            System.out.printf("Correct rate: %02.2f. Round saved, next LR: %f.%n", correctRate, learningRate);
        }

        final long end = System.currentTimeMillis();
        System.out.printf("All round finished, time: %s%n", Formats.millisDuration(end - start));
        executor.shutdown();
    }
}
