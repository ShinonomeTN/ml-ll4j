package com.shinonometn.tools;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Iterator;

/**
 * MNIST database dataset tools
 * <p>
 *
 * Handling MNIST dataset file format.
 * See: <a href="https://github.com/cvdfoundation/mnist?tab=readme-ov-file">GitHub - cvdfoundation/mnist</a>
 */
public final class MNIST {
    private MNIST() {
    }

    //================================================================

    public static EntryIterator createDataSetIterator(Path filePath) throws IOException {
        final InputStream inputStream = Files.newInputStream(filePath);

        // Read the magic
        final int[] magic = new int[4];
        for (int i = 0; i < magic.length; i++) {
            magic[i] = inputStream.read();
        }

        // Check the magic number
        final int magicSum = magic[0] + magic[1];
        if (magic[0] + magic[1] != 0) throw new IllegalArgumentException("Invalid magic sum of file: " + magicSum);

        // Read dimension sizes
        final int[] sizes = new int[magic[3]];
        for (int i = 0; i < sizes.length; i++) {
            sizes[i] = nextInt(inputStream, 4);
        }

        return new EntryIterator(getType(magic[2]), sizes, inputStream);
    }

    public static class EntryIterator implements Closeable, Iterator<Entry> {
        /** File data unit type */
        public final DataType type;
        /** Total dimension of this file */
        public final int dataSetDimensions;
        /** Entry count described in file meta, same as the size of the first dimension. */
        public final int entryCount;
        /** Entry data length */
        final int entrySize;

        private final int[] dimensionSizes;
        private final InputStream in;

        EntryIterator(final DataType type, final int[] dimensionSizes, final InputStream in) {
            this.type = type;
            this.dataSetDimensions = dimensionSizes.length;

            this.entrySize = (dataSetDimensions > 1)
                    ? Arrays.stream(dimensionSizes)
                            .skip(1)
                            .map(s -> s * type.size)
                            .reduce(1, (a, b) -> a * b)
                    : type.size; /* size * 1 */

            if (dataSetDimensions > 1) {
                this.dimensionSizes = new int[dataSetDimensions - 1];
                System.arraycopy(dimensionSizes, 1, this.dimensionSizes, 0, dataSetDimensions - 1);
            } else {
                this.dimensionSizes = new int[]{type.size};
            }

            this.in = in;
            this.entryCount = dimensionSizes[0];
        }

        public int getEntryDimensionSize(final int index) {
            return dimensionSizes[index];
        }

        public int getEntryDimensions() {
            return dimensionSizes.length;
        }

        /** Get the product of entry dimensions */
        public int getEntrySize() {
            return Arrays.stream(dimensionSizes).reduce(1, (a, b) -> a * b);
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

        private int yieldCount = 0;

        @Override
        public boolean hasNext() {
            return yieldCount < entryCount;
        }

        @Override
        public Entry next() {
            try {
                final byte[] data = new byte[entrySize];
                final int read = in.read(data, 0, entrySize);
                if (read != entrySize) throw new EOFException("Unexpected end of source stream");
                return new Entry(dimensionSizes, data);
            } catch (Exception e) {
                /* Throw all unexpected exceptions. */
                throw new RuntimeException(e);
            } finally {
                yieldCount++;
            }
        }
    }

    public static class Entry {
        private final int[] sizes;
        public final byte[] data;

        /** How many dimensions of this entry */
        public int getDimensions() {
            return sizes.length;
        }

        /** Product of dimensions */
        public int getDimensionSize(final int index) {
            return sizes[index];
        }

        Entry(final int[] sizes, final byte[] data) {
            this.sizes = sizes;
            this.data = data;
        }
    }

    //================================================================

    public enum DataType {
        /* 0x08 */ UByte("unsigned byte", 1, 0x08),
        /* 0x09 */ Byte("signed byte", 1, 0x09),
        /* 0x0A */ Reserved("reserved (unused)", -1, 0x0A),
        /* 0x0B */ Short("short (2 bytes)", 2, 0x0B),
        /* 0x0C */ Int("int (4 bytes)", 4, 0x0C),
        /* 0x0D */ Float("float (4 bytes)", 4, 0x0D),
        /* 0x0E */ Double("double (8 bytes)", 8, 0x0E);

        public final String description;
        public final int size;
        public final int code;

        DataType(final String description, final int size, int code) {
            this.description = description;
            this.size = size;
            this.code = code;
        }
    }

    static DataType getType(final int type) {
        final int index = type - 8;
        if (index >= DataType.values().length) throw new IllegalArgumentException("Invalid type code: " + type);
        return DataType.values()[index];
    }

    //================================================================

    /** Read next int from input stream */
    private static int nextInt(final InputStream is, final int bytes) throws IOException {
        int result = 0;
        for (int i = 0; i < bytes; i++) {
            result <<= 8;
            result |= (is.read() & 0xFF);
        }
        return result;
    }
}
