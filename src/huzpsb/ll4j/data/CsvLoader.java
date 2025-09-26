package huzpsb.ll4j.data;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Scanner;

public class CsvLoader {
    public static DataSet load(String path, int labelIndex) {
        try {
            final DataSet data = new DataSet();
            final Scanner sc = new Scanner(Files.newInputStream(Paths.get(path)));
            final String[] header = sc.nextLine().split(",");
            int n = header.length;
            while (sc.hasNextLine()) {
                final String[] line = sc.nextLine().split(",");
                final double[] x = new double[n - 1];
                final int label = Integer.parseInt(line[labelIndex]);
                int idx = 0;
                for (int i = 0; i < n; i++) {
                    if (i == labelIndex) continue;
                    x[idx++] = Double.parseDouble(line[i]);
                }
                final DataEntry entry = new DataEntry(label, x);
                data.split.add(entry);
            }
            return data;
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
