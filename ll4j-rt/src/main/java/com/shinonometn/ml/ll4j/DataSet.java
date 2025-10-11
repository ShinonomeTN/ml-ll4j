package com.shinonometn.ml.ll4j;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.Scanner;

public final class DataSet {
    private DataSet() {
    }

    public final static class LabeledEntry {
        public final int label;
        public final double[] values;

        LabeledEntry(final int label, final double[] values) {
            this.label = label;
            this.values = values;
        }
    }

    public interface SampleIterator<T> extends Iterator<T>, Closeable {
    }

    //================================================================

    public static SampleIterator<LabeledEntry> createCSVSampleIterator(String path) throws IOException {
        return createCSVSampleIterator(path, false);
    }

    public static SampleIterator<LabeledEntry> createCSVSampleIterator(String path, boolean skipHeader) throws IOException {
        final Scanner scanner = new Scanner(Files.newInputStream(Paths.get(path)));

        if (skipHeader) {
            final String header = scanner.nextLine().trim(); // Skip the csv header
            System.out.println("Header: " + header);
        }

        return new SampleIterator<LabeledEntry>() {
            @Override
            public boolean hasNext() {
                return scanner.hasNextLine();
            }

            @Override
            public LabeledEntry next() {
                final String[] line = scanner.nextLine().split(",");
                final double[] buffer = new double[line.length - 1];
                for (int i = 1; i < line.length; i++) buffer[i - 1] = Double.parseDouble(line[i]);
                final int label = Integer.parseInt(line[0]);
                return new LabeledEntry(label, buffer);
            }

            @Override
            public void close() throws IOException {
                scanner.close();
            }
        };
    }
}
