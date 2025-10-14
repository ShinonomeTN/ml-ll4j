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

    //================================================================

    /** A DataSet entry contains the value, and a label */
    public abstract static class Entry {
        public final double[] values;

        /** Get the label value array */
        abstract public double[] getLabel();

        public int getLabelSize() {
            return getLabel().length;
        }

        protected Entry(final double[] values) {
            this.values = values;
        }

        /** Set value data to an array */
        public void setValueTo(double[] dest) {
            System.arraycopy(getLabel(), 0, dest, 0, dest.length);
        }
    }

    //================================================================

    public interface SampleIterator<T> extends Iterator<T>, Closeable {
    }

    //================================================================

    /** Entry with a single label */
    public final static class LabelEntry extends Entry {

        private final double[] label;

        public double[] getLabel() {
            return label;
        }

        public int getLabelValue() {
            return (int) label[0];
        }

        LabelEntry(final int label, final double[] values) {
            super(values);
            this.label = new double[] { label };
        }

        @Override
        public int getLabelSize() {
            return label.length;
        }

        //================================================================

        public static SampleIterator<LabelEntry> createCSVIterator(String path) throws IOException {
            return createCSVIterator(path, true);
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
}
