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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.shinonometn.ml.ll4j.AdjustFunctions.fillWithGaussianRandom;
import static com.shinonometn.ml.ll4j.Layers.*;

public class FashionMnistTrain {
    private final static String LabeledDataPath = "fashion-mnist_train.csv";
    private final static String ModelLocation = "./test2.model";

    private static final ExecutorService executor = Executors.newSingleThreadExecutor();
    private static final Thread.UncaughtExceptionHandler uceHandler = (t, e) -> executor.execute(() -> {
        e.printStackTrace(System.err);
        executor.shutdown();
    });

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
        for (int i = 0; i < 128; i++) {
            final long roundStart = System.currentTimeMillis();
            System.out.printf("Training round % 3d/128 ========\n", i + 1);

            try(final SampleIterator<LabelEntry> sampleDataSet =
                        LabelEntry.createCSVIterator(LabeledDataPath, true)) {

                while(sampleDataSet.hasNext()) {
                    trainer.adjust(sampleDataSet.next(), 8e-7);

                    final int c = trainer.getIterationCount();
                    if ((c % 1000) != 0) continue;

                    final int t = trainer.getCorrectCount();
                    final int f = trainer.getWrongCount();
                    executor.execute(() -> System.out.printf(
                            "\r[% 6d] t:% 6d, f:% 6d, r:%2.2f%%",
                            c, t, f, ((t / (double) c) * 100)
                    ));
                }
            }

            System.out.println();

            final long roundEnd = System.currentTimeMillis();
            final int round = i;

            executor.execute(() -> {
                System.out.printf(
                        "Round %03d finished, time: %s%n",
                        round + 1, Formats.millisDuration(roundEnd - roundStart)
                );
                try {
                    trainer.writeModelToFile(ModelLocation);
                    System.out.println("Round saved.");
                } catch (IOException e) {
                    e.printStackTrace(System.err);
                }
            });
            trainer.resetCounters();
        }

        final long end = System.currentTimeMillis();
        System.out.printf("All round finished, time: %s%n", Formats.millisDuration(end - start));
        executor.shutdown();
    }
}
