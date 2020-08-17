package graph;

import com.google.common.collect.Sets;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static java.lang.System.exit;
//import static java.lang.System.out;

// Integrates graphs of different predicate valency
public class Valence {

    // Credit: https://stackoverflow.com/questions/852665/command-line-progress-bar-in-java
    private static void printProgress(int remain, int total, String message) {
        if (remain > total) {
            throw new IllegalArgumentException();
        }
        int maxBareSize = 10; // 10unit for 100%
        int remainProcent = ((100 * remain) / total) / maxBareSize;
        char defaultChar = '-';
        String icon = "#";
        String bare = new String(new char[maxBareSize]).replace('\0', defaultChar) + "]";
        StringBuilder bareDone = new StringBuilder();
        bareDone.append("[");
        for (int i = 0; i < remainProcent; i++) {
            bareDone.append(icon);
        }
        String bareRemain = bare.substring(remainProcent, bare.length());

        int maxMsgLen = 60;
        int msgLen = Math.min(maxMsgLen, message.length());
        StringBuilder sb = new StringBuilder();
        sb.append(message, 0, msgLen);
        for (int i = msgLen; i < maxMsgLen; i++) { sb.append(" "); }

        System.out.print("\r" + bareDone + bareRemain + " " + remainProcent * 10 + "% : " + sb.toString());
        if (remain == total) {
            System.out.print("\n");
        }
    }

    // Integrate an argwise graph and binary graph in two-typed space
    private static void integrateBinaryGraphPair(PGraph graphA, PGraph graphB, Path dirDest, int topKUnaries) throws IOException {
        // Initialize file writer
        PrintWriter output = new PrintWriter(new BufferedWriter(new FileWriter(dirDest.toString())));

        // Write file header
        output.println("types: " + graphB.types + ", num preds: " + graphB.nodes.size());

        // Initialize metadata structures
        String[] graphTypes = graphB.types.split("#");
        Map<String, String> typeToOtherType = new HashMap<>();
        typeToOtherType.put(graphTypes[0], graphTypes[1]);
        typeToOtherType.put(graphTypes[1], graphTypes[0]);
        Set<String> entailedUnaries = new HashSet<>();

        // Write out nodes from the binary graph
        for (Node node : graphB.nodes) {
            String pred = node.id;

            String[] argIDs = {"[arg1]", "[arg2]"};
            String[] argwisePreds = {argIDs[0] + pred, argIDs[1] + pred};

            // Determine if node has a nonzero number of edges
            int numBinaryEntailments = node.oedges.size();
            int numArg1Entailments = graphA.pred2node.containsKey(argwisePreds[0]) ? graphA.pred2node.get(argwisePreds[0]).oedges.size() : 0;
            int numArg2Entailments = graphA.pred2node.containsKey(argwisePreds[1]) ? graphA.pred2node.get(argwisePreds[1]).oedges.size() : 0;
            int totalEntailments = numBinaryEntailments + numArg1Entailments + numArg2Entailments;
//            if (totalEntailments == 0) {
//                continue;
//            }

            // Write node header
            output.println("predicate: " + pred);
            output.println("num neighbors: " + totalEntailments);
            output.println();
            output.println("BInc sims");

            // Write B->B entailments
            if (numBinaryEntailments > 0) {
                writeEntailments("", "", node, graphB, output);
            }

            // Write B->U entailments (convert [sub]->[arg1], [obj]->[arg2])
            if (numArg1Entailments > 0 || numArg2Entailments > 0) {

                // (in the case of type-symmetric binary graphs we need to add appropriate _1 and _2 argument labels to unaries)
                String[] predTypes = pred.substring(pred.indexOf("#")+1).split("#");
                boolean symmetricGraphTyping = graphTypes[0].equals(graphTypes[1]);
                String[] symmetricTypeIDs = {"", ""};
                if (symmetricGraphTyping) {
                    symmetricTypeIDs = new String[]{
                            "_" + predTypes[0].substring(predTypes[0].length() - 1),
                            "_" + predTypes[1].substring(predTypes[1].length() - 1)
                    };
                }

                double minSim = 0.02;

                for (int i : new Integer[]{0, 1}) {
                    String argwisePred = argwisePreds[i];
                    Node argwiseNode = graphA.pred2node.get(argwisePred);
                    if (argwiseNode == null) { continue; }
                    List<String> entailedPreds = argwiseNode.oedges.stream().map(e -> graphA.idx2node.get(e.nIdx).id).collect(Collectors.toList());
                    entailedUnaries.addAll(entailedPreds);

                    String unaryType = argwisePred.split("#")[i+1];
                    String basicType = symmetricGraphTyping ? unaryType.substring(0, unaryType.length() - 2) : unaryType;
                    String typeSuffix = typeToOtherType.get(basicType);
                    typeSuffix = symmetricTypeIDs[i] + "#" + typeSuffix + symmetricTypeIDs[(i+1)%2];
                    String argIDTag;
                    if (symmetricGraphTyping) {
                        argIDTag = unaryType.substring(unaryType.length() - 1);
                    } else {
                        int typeIndex = Arrays.asList(graphTypes).indexOf(basicType)+1;
                        argIDTag = Integer.toString(typeIndex);
                    }
                    String prefixTag = "[gtype" + argIDTag + "]";
                    writeEntailments(prefixTag, typeSuffix, argwiseNode, graphA, output, minSim, topKUnaries);
                }
            }

            output.println();
            output.println();
        }

        for (String entailedUnary : entailedUnaries) {
            // Write node header
            output.println("predicate: " + entailedUnary);
            output.println("num neighbors: " + 0);
            output.println();
            output.println();
        }

        output.flush();
        output.close();
    }

    private static void makeDummyAugmentedUnaryGraphs(Path dirUnaryIn, Path dirUnaryOut, int topKUnaries) throws IOException {
        Set<String> filenames = filenamesFromDirectory(dirUnaryIn, "^[^#]*_sim\\.txt");
        File outputFolder = new File(dirUnaryOut.toString());
        if (!outputFolder.exists()) {
            outputFolder.mkdir();
        }

        String dummySuffix = "#unary";

        int numGraphs = filenames.size();
        int progress = 0;
        for (String fname : filenames) {
            printProgress(progress, numGraphs, fname);

            // Read graph
            PGraph graph = new PGraph(dirUnaryIn.resolve(fname).toString());

            // Initialize file writer
            String dummyFname = fname.replace("_sim", dummySuffix + "_sim");
            PrintWriter output = new PrintWriter(new BufferedWriter(new FileWriter(dirUnaryOut.resolve(dummyFname).toString())));

            // Write file header
            output.println("types: " + graph.types + dummySuffix + ", num preds: " + graph.nodes.size());

            for (Node node : graph.nodes) {
                String pred = node.id;
                int numEntailments = node.oedges.size();

                // Write node header
                output.println("predicate: " + removeUnaryTag(pred) + dummySuffix);
                output.println("num neighbors: " + numEntailments);
                output.println();
                output.println("BInc sims");

                writeEntailments("", dummySuffix, node, graph, output, 0.01, topKUnaries);

//                if (node.oedges.size() == 0) {
//                    System.out.println(String.format("GOOD! pred (%s) has no edges but we wrote it in graph nodes", node.id));
//                    exit(1);
//                }

                output.println();
                output.println();
            }

            output.flush();
            output.close();
            progress++;
        }

        printProgress(numGraphs, numGraphs, "done");
        System.out.println();
    }

    private static String removeUnaryTag(String pred) {
        return pred.replaceFirst("\\[.*?\\]", "");
    }

    private static void writeEntailments(String prefix, String suffix, Node node, PGraph graph, PrintWriter output, double minSim, int topKUnaries) {
        int ct = 0;
        for (Oedge edge : node.oedges) {
            float score = edge.sim;

            String entailedPred = graph.idx2node.get(edge.nIdx).id;

            boolean unaryPred = (entailedPred.split("#").length == 2);
            if (unaryPred) {
                if (score < minSim) { continue; }
                if (topKUnaries >= 0 && ct >= topKUnaries) {
                    break;
                }
                ct++;
                entailedPred = removeUnaryTag(entailedPred);
            }

            output.println(prefix + entailedPred + suffix + " " + score);
        }
    }

    private static void writeEntailments(String prefix, String suffix, Node node, PGraph graph, PrintWriter output, double minSim) {
        writeEntailments(prefix, suffix, node, graph, output, minSim, -1);
    }

    private static void writeEntailments(String prefix, String suffix, Node node, PGraph graph, PrintWriter output) {
        writeEntailments(prefix, suffix, node, graph, output, 0.0);
    }

    private static void integrateEntailmentGraphs(Path dirArgwise, Path dirBinary, Path dirDest, boolean checkOnly, int topKUnaries) {
        // Fetch filenames for each graph in the given directories
        Set<String> filenamesArgwise = filenamesFromDirectory(dirArgwise, ".*_sim\\.txt");
        System.out.println("found argwise graphs: " + dirArgwise.toString());
        Set<String> filenamesBinary = filenamesFromDirectory(dirBinary, ".*_sim\\.txt");
        System.out.println("found binary graphs: " + dirBinary.toString());

        // Identify overlapping graphs (primary case), unary-only graphs, and binary-only graphs
        Set<String> filenamesIntersection = Sets.intersection(filenamesArgwise, filenamesBinary);
        filenamesArgwise = Sets.difference(filenamesArgwise, filenamesIntersection);
        filenamesBinary = Sets.difference(filenamesBinary, filenamesIntersection);

        if (filenamesIntersection.size() == 0) {
            System.err.println("No common graphs between binary and argwise sets");
            exit(1);
        }

        System.out.println("Intersection of binary and argwise graphs: " + filenamesIntersection.size());
        System.out.println("Argwise-only graphs: " + filenamesArgwise.size());
        System.out.println("Binary-only graphs: " + filenamesBinary.size());

        if (checkOnly) {
            exit(0);
        }

        File destFolder = new File(dirDest.toString());
        if (!destFolder.exists()) {
            destFolder.mkdir();
        }

        int numGraphs = filenamesIntersection.size();
        int progress = 0;
        int aGraphsSkipped = 0;
        int bGraphsSkipped = 0;

        for (String fname : filenamesIntersection) {
            printProgress(progress, numGraphs, fname);
            progress++;
            Path destFname = destFolder.toPath().resolve(fname);
            try {
                PGraph graphA = new PGraph(dirArgwise.resolve(fname).toString());
                PGraph graphB = new PGraph(dirBinary.resolve(fname).toString());
                if (graphA.name == null) {
//                    System.err.println("Empty graphA: " + fname);
                    aGraphsSkipped++;
                }
                if (graphB.name == null) {
//                    System.err.println("Empty graphB: " + fname);
                    bGraphsSkipped++;
                }
                if (graphA.name == null || graphB.name == null) {
                    continue;
                }
                integrateBinaryGraphPair(graphA, graphB, destFname, topKUnaries);
            } catch (IOException e) {
                System.err.println("IOException: " + fname);
                System.err.println(e.getMessage());
                e.printStackTrace();
            }
        }

        printProgress(numGraphs, numGraphs, "done");
        System.out.println();

        if (aGraphsSkipped > 0 || bGraphsSkipped > 0) {
            System.err.println("Skipped " + aGraphsSkipped + " argwise graphs");
            System.err.println("Skipped " + bGraphsSkipped + " binary graphs");
        }

//        for (String fname : filenamesUn) {
//            writeOutGraph(dirUn.resolve(fname), dirDest);
//        }
//
//        for (String fname : filenamesBi) {
//            writeOutGraph(dirBi.resolve(fname), dirDest);
//        }
    }

    // Returns a set of filenames of each graph in the given directory
    public static Set<String> filenamesFromDirectory(Path directory, String matchRex) {
        File[] files = directory.toFile().listFiles((dir, name) -> name.matches(matchRex));
        Set<String> ret = Arrays.stream(files).map(File::getName).collect(Collectors.toSet());
        return ret;
    }

    public static void main(String[] args) throws IOException {
        // *** EXPECTED PROGRAM ARGS
        // 0 Destination folder for integrated graphs
        // 1 Source folder for unary/argwise graphs
        // 2 Source folder for binary graphs
        // *** EXPECTED CONDITIONS
        // - only BInc scores will be written out, so no other sims scores are needed in the input
        List<String> argList = new ArrayList<>(Arrays.asList(args));

        // Print stats on which graphs will be integrated
        if (argList.remove("--merge")) {
            boolean checkOnly = false;
            if (argList.remove("--check")) {
                checkOnly = true;
                System.out.print("[CHECK-ONLY] ");
            }

            if (argList.size() < 3) {
                System.out.println("Usage: --merge [--check] <destination-folder> <argwise-folder> <binary-folder> [topKUnaries]");
                exit(1);
            }

            System.out.println("Valence: MERGE ARGWISE AND BINARY GRAPHS");

            Path dirDest = Paths.get(argList.get(0));
            Path dirArgwise = Paths.get(argList.get(1));
            Path dirBinary = Paths.get(argList.get(2));

            int topKUnaries = -1;

            if (argList.size() == 4) {
                topKUnaries = Integer.parseInt(argList.get(3));
                if (topKUnaries % 2 == 1) {
                    System.err.println("Given topKUnaries: " + topKUnaries + ". Parameter must be an even number as it is divided evenly between arg1 and arg2.");
                    exit(1);
                }
                System.out.println("* integrating with top " + topKUnaries + " unaries");
            }

            // Integrate the graphs into one set of multi-valency graphs
            integrateEntailmentGraphs(dirArgwise, dirBinary, dirDest, checkOnly, topKUnaries);
        }

        else if (argList.remove("--makeDummy")) {
            if (argList.size() < 2) {
                System.out.println("Usage: Valence --makeDummy <unary-in-folder> <unary-out-folder> [keep-top-K]");
                exit(1);
            }

            System.out.println("Valence: MAKE UNARY DUMMY-TYPE GRAPHS");

            Path dirUnaryIn = Paths.get(argList.get(0));
            Path dirUnaryOut = Paths.get(argList.get(1));

            int topKUnaries = argList.size() == 3 ? Integer.parseInt(argList.get(2)) : -1;

            makeDummyAugmentedUnaryGraphs(dirUnaryIn, dirUnaryOut, topKUnaries);
        }

        else {
            System.out.println("Available programs:");
            System.out.println("Merge:\t\t--merge");
            System.out.println("MakeDummy:\t--makeDummy");
        }

    }
}
