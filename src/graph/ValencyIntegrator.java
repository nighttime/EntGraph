package graph;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

// Integrates graphs of different predicate valency
public class ValencyIntegrator {

    // Integrate an argwise graph and binary graph in two-typed space
    public static void integrateBinaryGraphPair(Path pathAGraph, Path pathBGraph, Path dirDest) {
//        List<String> linesUn;
//        List<String> linesBi;
//        try {
//            linesUn = Files.readAllLines(pathUGraph);
//        } catch (IOException e) {
//            System.err.println("Could not read unary file: " + e.getMessage());
//            return;
//        }
//        try {
//            linesBi = Files.readAllLines(pathUGraph);
//        } catch (IOException e) {
//            System.err.println("Could not read binary file: " + e.getMessage());
//            return;
//        }

        PGraph graphA = new PGraph(pathAGraph.toString());
        PGraph graphB = new PGraph(pathBGraph.toString());

        // Write file header

        // Start with existing nodes in the binary graph
        for (Node node : graphB.nodes) {
            String pred = node.id;
            // Write header for this node

            // Write header for binary -> binary entailments

            writeEntailments(node, graphB);

            // Write header for argwise -> argwise entailments

            String[] argwisePreds = {"[sub]" + pred, "[obj]" + pred};
            for (String argwisePred : argwisePreds) {
                if (graphA.pred2node.containsKey(argwisePred)) {
                    Node argwiseNode = graphA.pred2node.get(argwisePred);
                    writeEntailments(argwiseNode, graphA);
                }
            }
        }
    }

    public static void writeEntailments(Node node, PGraph graph) {
        for (Oedge edge : node.oedges) {
            String entailedPred = graph.idx2node.get(edge.nIdx).id;
            float score = edge.sim;
            // Write entailment
        }
    }

    public static void integrateEntailmentGraphs(Path dirArgwise, Path dirBinary, Path dirDest) {
        // Fetch filenames for each graph in the given directories
        Set<String> filenamesArgwise = graphFilenamesFromDirectory(dirArgwise);
        Set<String> filenamesBinary = graphFilenamesFromDirectory(dirBinary);

        // Identify overlapping graphs (primary case), unary-only graphs, and binary-only graphs
        Set<String> intersection = new HashSet<>(filenamesArgwise);
        intersection.retainAll(filenamesBinary);
        filenamesArgwise.removeAll(intersection);
        filenamesBinary.removeAll(intersection);

        for (String fname : intersection) {
            fname = "organization#location_sim.txt";
            integrateBinaryGraphPair(dirArgwise.resolve(fname), dirBinary.resolve(fname), dirDest);
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
        return Arrays.stream(files).map(File::getName).collect(Collectors.toSet());
    }

    public static void main(String[] args) {
        // EXPECTED PROGRAM ARGS
        // 1 Destination folder for integrated graphs
        // 2 Source folder for unary/argwise graphs
        // 3 Source folder for binary graphs
        if (args.length != 3) {
            args = new String[]{"newsspike_sims/newsspike_multivalent", "newsspike_sims/newsspike_argwise_500k_argnumbers_3_3", "newsspike_sims/newsspike_argwise_500k_argnumbers_3_3"};
            System.err.println("Using default program args: " + Arrays.toString(args));
        }

        Path dirDest = Paths.get(args[0]);
        Path dirArgwise = Paths.get(args[1]);
        Path dirBinary = Paths.get(args[2]);

        // Integrate the graphs into one set of multi-valency graphs
        integrateEntailmentGraphs(dirArgwise, dirBinary, dirDest);
    }
}
