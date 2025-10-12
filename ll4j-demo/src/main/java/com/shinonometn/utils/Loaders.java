package com.shinonometn.utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public final class Loaders {
    private Loaders() {}

    public static String[] loadModelString(final String modelPath) throws IOException {
        final List<String> buffer = new ArrayList<>();
        try (final Scanner scanner = new Scanner(Files.newInputStream(Paths.get(modelPath)))) {
            while (scanner.hasNextLine()) buffer.add(scanner.nextLine());
        }
        return buffer.toArray(new String[0]);
    }
}
