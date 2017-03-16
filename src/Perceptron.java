/**
 * Created by Hao Xiong on 3/16/2017.
 * Copyright belongs to Hao Xiong, Email: haoxiong@outlook.com
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

public class Perceptron {

    private static final String[] STOP_WORDS = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"};
    public static final int HAM = 0;
    public static final int SPAM = 1;
    private static final String[] TRAIN_PATH = {"dataset/enron4/train/ham", "dataset/enron4/train/spam"};
    private static final String[] TEST_PATH = {"dataset/enron4/test/ham", "dataset/enron4/test/spam"};

    private static List<String> stop_words;
    private static Map<String, Integer> dictionary;
    private static double[] W;
    private static int[] numberOfTestFiles = {0, 0};

    //static initializer
    static {
        numberOfTestFiles[HAM] = new File(TEST_PATH[HAM]).list().length;
        numberOfTestFiles[SPAM] = new File(TEST_PATH[SPAM]).list().length;
        stop_words = Arrays.asList(STOP_WORDS);
    }

    public static void main(String[] args) {
        double learningRate = 0.0071;
        int iteration = 70;
        boolean removeStopWords = false;
        countWords(TRAIN_PATH, removeStopWords);
        trainPerceptron(toVectors(TRAIN_PATH, removeStopWords), learningRate, iteration);
        System.out.println("learningRate = " + learningRate + ", iteration = " + iteration + " => Accuracy: " + testAccuracy(toVectors(TEST_PATH, removeStopWords)));
    }

    /**
     * Record the frequency of each word in the given training text.
     */
    private static void countWords(String[] folders, boolean filter) {
        dictionary = new LinkedHashMap<>();
        for (String folder : folders) {
            try (Stream<Path> paths = Files.walk(Paths.get(folder))) {
                paths.forEach(filePath -> {
                    if (Files.isRegularFile(filePath)) {
                        String line;
                        try (BufferedReader br = new BufferedReader(new FileReader(filePath.toFile()))) {
                            while ((line = br.readLine()) != null) {
                                String[] words = line.split(" ");
                                for (String word : words) {
                                    if (filter && stop_words.contains(word)) continue;
                                    //put the word into dictionary and count the number of each word
                                    int num = dictionary.containsKey(word) ? dictionary.get(word) + 1 : 1;
                                    dictionary.put(word, num);
                                }
                            }
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                });
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * This method and its sub method are used to convert the raw data to vectors
     * Can only be called after dictionary has been constructed
     */
    private static ArrayList<TextVector> toVectors(String[] folders, boolean filter) {
        ArrayList<TextVector> vectors = new ArrayList<>();
        List<String> word_list = new ArrayList<>(dictionary.keySet());
        fillVector(folders, HAM, word_list, vectors, filter);
        fillVector(folders, SPAM, word_list, vectors, filter);
        return vectors;
    }

    private static void fillVector(String[] folders, int type, List<String> word_list, ArrayList<TextVector> vectors, boolean filter) {
        try (Stream<Path> paths = Files.walk(Paths.get(folders[type]))) {
            paths.forEach(filePath -> {
                if (Files.isRegularFile(filePath)) {
                    String line;
                    int[] features = new int[word_list.size() + 1];
                    features[0] = 1;    // set x0=1 for all vectorNormalize X
                    try (BufferedReader br = new BufferedReader(new FileReader(filePath.toFile()))) {
                        while ((line = br.readLine()) != null) {
                            String[] words = line.split(" ");
                            for (String word : words) {
                                if (filter && stop_words.contains(word)) continue;
                                int index = word_list.indexOf(word) + 1;
                                if (index != 0) features[index]++;
                            }
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    vectors.add(new TextVector(features, type == HAM ? -1 : 1));
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Implement the perceptron algorithm by using the perceptron training rule
     *
     * @param learningRate the learning rate in gradient ascent
     */
    private static void trainPerceptron(ArrayList<TextVector> vectors, double learningRate, int repeat) {
        int size = dictionary.size() + 1;
        W = new double[size];//initially set all w=0
        while (repeat-- > 0) {
            for (TextVector tv : vectors) {
                int error = tv.predictionError(W);
                if (error != 0) {
                    for (int i = 0; i < size; i++) {
                        //update the parameters according to prediction error with respect to this single training example only.
                        if (tv.features[i] != 0) W[i] += learningRate * tv.features[i] * error;
                    }
                }
            }
        }
    }

    /**
     * Test the accuracy of perceptron algorithm
     */
    private static double testAccuracy(ArrayList<TextVector> test_vectors) {
        int correct = 0;
        for (TextVector tv : test_vectors) {
            if (tv.predictionError(W) == 0) correct++;
        }
        return (double) correct / (numberOfTestFiles[HAM] + numberOfTestFiles[SPAM]);
    }
}

/**
 * Represent each example vector
 */
class TextVector {
    //key: the index of the word in dictionary, value the number of the word
    public int[] features;
    public int type;

    TextVector(int[] fts, int tp) {
        features = fts;
        type = tp;
    }

    //namely (t - o)
    int predictionError(double[] W) {
        return type - estimateType(W);
    }

    private int estimateType(double[] W) {
        double sum = 0;
        for (int i = 0; i < W.length; i++) {
            if (features[i] != 0) sum += W[i] * features[i];
        }
        return sum > 0 ? 1 : -1;
    }
}