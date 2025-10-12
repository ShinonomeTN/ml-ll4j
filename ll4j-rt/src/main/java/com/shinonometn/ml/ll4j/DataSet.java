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

    public abstract static class Entry {
        public final double[] values;

        abstract public int getLabelLength();

        protected Entry(final double[] values) {
            this.values = values;
        }

        abstract public void toValues(double[] input);
    }

    public final static class LabelEntry extends Entry {
        public final int label;

        LabelEntry(final int label, final double[] values) {
            super(values);
            this.label = label;
        }

        @Override
        public int getLabelLength() {
            return 1;
        }

        @Override
        public void toValues(final double[] input) {
            input[0] = label;
        }

        //================================================================

        public static SampleIterator<LabelEntry> createCSVIterator(String path) throws IOException {
            return createCSVIterator(path, false);
        }

        public static SampleIterator<LabelEntry> createCSVIterator(String path, boolean skipHeader) throws IOException {
            final Scanner scanner = new Scanner(Files.newInputStream(Paths.get(path)));

            if (skipHeader) scanner.nextLine(); // Skip the csv header

            return new SampleIterator<LabelEntry>() {
                @Override
                public boolean hasNext() {
                    return scanner.hasNextLine();
                }

                @Override
                public LabelEntry next() {
                    final String[] line = scanner.nextLine().split(",");
                    final double[] buffer = new double[line.length - 1];
                    for (int i = 1; i < line.length; i++) buffer[i - 1] = Double.parseDouble(line[i]);
                    final int label = Integer.parseInt(line[0]);
                    return new LabelEntry(label, buffer);
                }

                @Override
                public void close() {
                    scanner.close();
                }
            };
        }
    }

    public interface SampleIterator<T> extends Iterator<T>, Closeable {
    }
}
