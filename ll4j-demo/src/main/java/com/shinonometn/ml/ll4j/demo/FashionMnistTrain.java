package com.shinonometn.ml.ll4j.demo;

import com.shinonometn.ml.ll4j.DataSet;
import com.shinonometn.ml.ll4j.ModelTrainer;
import com.shinonometn.utils.Formats;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.shinonometn.ml.ll4j.AdjustFunctions.fillWithGaussianRandom;
import static com.shinonometn.ml.ll4j.Layers.*;

public class FashionMnistTrain {
    private final static String LabeledDataPath = "fashion-mnist_train.csv";
    private final static String ModelPath = "./test2.model";

    private static final ExecutorService executor = Executors.newSingleThreadExecutor();
    private static final Thread.UncaughtExceptionHandler uceHandler = (t, e) -> executor.execute(() -> {
        e.printStackTrace(System.err);
        executor.shutdown();
    });

    public static void main(String[] args) throws IOException {
        Thread.currentThread().setUncaughtExceptionHandler(uceHandler);

        System.out.printf("Training data file : %s\n", Paths.get(LabeledDataPath).toAbsolutePath());
        System.out.printf("Model output file  : %s\n", Paths.get(ModelPath).toAbsolutePath());

        final ModelTrainer trainer = ModelTrainer.create(
                fillWithGaussianRandom(dense(784, 100)),
                leakyRelu(100),
                fillWithGaussianRandom(dense(100, 100)),
                leakyRelu(100),
                fillWithGaussianRandom(dense(100, 10)),
                judge(10)
        );

        final long start = System.currentTimeMillis();
        for (int i = 0; i < 128; i++) {
            final long roundStart = System.currentTimeMillis();
            System.out.printf("Training round %03d ========\n", i + 1);

            try(final DataSet.SampleIterator<DataSet.LabelEntry> sampleDataSet = DataSet.LabelEntry
                    .createCSVIterator(LabeledDataPath, true)) {

                while(sampleDataSet.hasNext()) {
                    trainer.adjust(sampleDataSet.next(), 8e-7);

                    final int c = trainer.getIterationCount();
                    if ((c % 1000) == 0) {
                        final int t = trainer.getCorrectCount();
                        final int f = trainer.getWrongCount();

                        executor.execute(() -> System.out.printf(
                                "[% 6d] t: % 4d, f: %4d, r: %.2f%%\r",
                                c, t, f, ((t / (double) c) * 100)
                        ));

                    }
                }
            }

            System.out.println();

            final long roundEnd = System.currentTimeMillis();
            final int round = i;

            executor.execute(() -> {
                System.out.printf(
                        "Round %03d finished, time: %s ========%n",
                        round + 1, Formats.millisDuration(roundEnd - roundStart)
                );
                try {
                    trainer.writeModelToFile(ModelPath);
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
