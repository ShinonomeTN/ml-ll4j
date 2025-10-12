package huzpsb.ll4j.samples;

import com.shinonometn.utils.Formats;
import huzpsb.ll4j.data.CsvLoader;
import huzpsb.ll4j.data.DataSet;
import huzpsb.ll4j.layer.DenseLayer;
import huzpsb.ll4j.layer.JudgeLayer;
import huzpsb.ll4j.layer.LeakyRelu;
import huzpsb.ll4j.model.Model;

import java.nio.file.Paths;
import java.time.Duration;
import java.time.chrono.Chronology;

public class TestTrain {
    private final static String ModelPath = "test3.model";
    private final static String LabeledDataPath = "fashion-mnist_train.csv";

    public static void main(String[] args) {
        System.out.printf("Training data file : %s\n", Paths.get(LabeledDataPath).toAbsolutePath());
        System.out.printf("Model output file  : %s\n", Paths.get(ModelPath).toAbsolutePath());

        DataSet trainingSet = CsvLoader.load(LabeledDataPath, 0);

        Model model = new Model(
                new DenseLayer(784, 100)
                , new LeakyRelu(100)
                , new DenseLayer(100, 100)
                , new LeakyRelu(100)
                , new DenseLayer(100, 10)
                , new JudgeLayer(10) // MSELoss
        );

        final long start = System.currentTimeMillis();
        for (int i = 0; i < 128; i++) {
            final long roundStart = System.currentTimeMillis();
            model.trainOn(trainingSet);
            final long roundEnd = System.currentTimeMillis();
            System.out.printf("Round %03d finished, time: %s ========%n", i + 1, Formats.millisDuration(roundEnd - roundStart));
            model.save(ModelPath);
        }

        final long end = System.currentTimeMillis();
        System.out.printf("All round finished, time: %s%n", Formats.millisDuration(end - start));
    }
}
