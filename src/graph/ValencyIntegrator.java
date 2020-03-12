package graph;

import com.google.common.collect.Sets;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static java.lang.System.exit;
import static java.lang.System.setErr;

// Integrates graphs of different predicate valency
public class ValencyIntegrator {

    // Integrate an argwise graph and binary graph in two-typed space
    public static void integrateBinaryGraphPair(Path pathAGraph, Path pathBGraph, Path dirDest) throws IOException {
        PGraph graphA = new PGraph(pathAGraph.toString());
        PGraph graphB = new PGraph(pathBGraph.toString());

        // Initialize file writer
        PrintWriter output = new PrintWriter(new BufferedWriter(new FileWriter(dirDest.toString())));

        // Write file header
        output.println("types: " + graphB.types + ", num preds: " + graphB.nodes.size());

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
            if (totalEntailments == 0) {
                continue;
            }

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
            // (in the case of type-symmetric binary graphs we need to add appropriate _1 and _2 argument labels to unaries)
            String[] graphTypes = graphB.types.split("#");
            boolean symmetricTypes = graphTypes[0].equals(graphTypes[1]);
            String[] predTypes = pred.substring(pred.indexOf("#")+1).split("#");

            if (numArg1Entailments > 0) {
                Node argwiseNode = graphA.pred2node.get(argwisePreds[0]);
                String suffix = symmetricTypes ? predTypes[0].substring(predTypes[0].indexOf("_")) : "";
                suffix += "#" + predTypes[1];
                writeEntailments(argIDs[0], suffix, argwiseNode, graphA, output);
            }

            if (numArg2Entailments > 0) {
                Node argwiseNode = graphA.pred2node.get(argwisePreds[1]);
                String suffix = symmetricTypes ? predTypes[1].substring(predTypes[1].indexOf("_")) : "";
                suffix += "#" + predTypes[0];
                writeEntailments(argIDs[1], suffix, argwiseNode, graphA, output);
            }

            output.println();
            output.println();
        }

        output.flush();
        output.close();
    }

    public static void writeEntailments(String prefix, String suffix, Node node, PGraph graph, PrintWriter output) {
        for (Oedge edge : node.oedges) {
            float score = edge.sim;
            if (score < 0.05) { continue; }

            String entailedPred = graph.idx2node.get(edge.nIdx).id + suffix;
            boolean unaryPred = entailedPred.startsWith("[unary]");
            if (unaryPred) {
                entailedPred = entailedPred.replaceFirst("\\[unary\\]", "");
            }

//            output.println(prefix + entailedPred + " " + score);
            output.println(entailedPred + " " + score);
        }
    }

    public static void integrateEntailmentGraphs(Path dirArgwise, Path dirBinary, Path dirDest) {
        // Fetch filenames for each graph in the given directories
        Set<String> filenamesArgwise = graphFilenamesFromDirectory(dirArgwise);
        Set<String> filenamesBinary = graphFilenamesFromDirectory(dirBinary);

        // Identify overlapping graphs (primary case), unary-only graphs, and binary-only graphs
        Set<String> filenamesIntersection = Sets.intersection(filenamesArgwise, filenamesBinary);
        filenamesArgwise = Sets.intersection(filenamesArgwise, filenamesIntersection);
        filenamesBinary = Sets.intersection(filenamesBinary, filenamesIntersection);

        if (filenamesIntersection.size() == 0) {
            System.err.println("No common graphs between binary and argwise sets");
            exit(1);
        }

        File destFolder = new File(dirDest.toString());
        destFolder.mkdir();

        for (String fname : filenamesIntersection) {
            Path destFname = destFolder.toPath().resolve(fname);
            try {
                integrateBinaryGraphPair(dirArgwise.resolve(fname), dirBinary.resolve(fname), destFname);
                System.out.println("Integrated: " + fname);
            } catch (IOException e) {
                System.err.println("IOException: " + fname);
                System.err.println(e.getMessage());
                e.printStackTrace();
            }
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
    public static Set<String> graphFilenamesFromDirectory(Path directory) {
        File[] files = directory.toFile().listFiles((dir, name) -> name.endsWith("_sim.txt"));
        Set<String> ret = Arrays.stream(files).map(File::getName).collect(Collectors.toSet());
        return ret;
    }

    public static void main(String[] args) {
        // *** EXPECTED PROGRAM ARGS
        // 0 Destination folder for integrated graphs
        // 1 Source folder for unary/argwise graphs
        // 2 Source folder for binary graphs
        // *** EXPECTED CONDITIONS
        // - only BInc scores will be written out, so no other sims scores are needed in the input

        if (args.length != 3) {
            args = new String[]{"newsspike_sims/multivalent_dummy", "newsspike_sims/newsspike_integrator_test_argwise", "newsspike_sims/newsspike_integrator_test_binary"};
            System.out.println("* Using default program args: " + Arrays.toString(args));
        } else {
            System.out.println("* Using given program args: " + Arrays.toString(args));
        }

        Path dirDest = Paths.get(args[0]);
        Path dirArgwise = Paths.get(args[1]);
        Path dirBinary = Paths.get(args[2]);

        // Integrate the graphs into one set of multi-valency graphs
        integrateEntailmentGraphs(dirArgwise, dirBinary, dirDest);
    }
}
