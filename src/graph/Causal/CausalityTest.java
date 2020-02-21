package graph.Causal;

import com.google.common.collect.Multiset;
import com.google.common.collect.Multisets;
import com.google.common.collect.Sets;
import com.google.common.collect.TreeMultiset;
import constants.ConstantsGraphs;
import graph.Edge;
import graph.Node;
import graph.Oedge;
import graph.PGraph;
import in.sivareddy.graphparser.util.graph.Graph;

import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.lang.Math.min;

public class CausalityTest {

	public static final String ANSI_RESET = "\u001B[0m";
	public static final String ANSI_BLACK = "\u001B[30m";
	public static final String ANSI_RED = "\u001B[31m";
	public static final String ANSI_GREEN = "\u001B[32m";
	public static final String ANSI_YELLOW = "\u001B[33m";
	public static final String ANSI_BLUE = "\u001B[34m";
	public static final String ANSI_PURPLE = "\u001B[35m";
	public static final String ANSI_CYAN = "\u001B[36m";
	public static final String ANSI_WHITE = "\u001B[37m";

	private static final DecimalFormat round = new DecimalFormat("#.000");

	public static void printPredicateEntailments(int topK) {
		PGraph.setPredToOcc(ConstantsGraphs.root);
		for (PGraph graph : GraphSet.generator()) {
			System.out.println(ANSI_YELLOW + graph.name + ANSI_RESET);
			System.out.println(graph.nodes.size());
			graph.setSortedEdges();
			for (Node n : graph.nodes) {
				System.out.println(ANSI_BLUE + n.id + ANSI_RESET + "\n--------");
				Map<Node, Float> entailmentMap = getEntailments(n, graph, false);
				printEntailmentMapSorted(n, entailmentMap, topK);
			}
			System.out.println("\n=========\n");
		}
	}

	public static void printCausalEntailments() {
		System.out.println("Reading occurrence files..");
		PGraph.setPredToOcc(ConstantsGraphs.root);
		System.out.println("Scanning graphs...");

		for (PGraph graph : GraphSet.generator()) {
			if (graph.nodes.size() == 0) { continue; }

			graph.setSortedEdges();
			Set<Integer> visitedNodes = new HashSet<>();
			for (Edge edge : graph.sortedEdges) {
				if (visitedNodes.contains(edge.i)) { continue; }
				visitedNodes.add(edge.i);

				Node base_node = graph.idx2node.get(edge.i);
				String base_pred = base_node.id;
				if (base_pred.contains("__") || base_pred.split("thing").length > 2) { continue; }

				/* String try_pred = "trying__" + base_pred;
				if (!graph.pred2node.containsKey(try_pred)) { continue; }
				Node try_node = graph.pred2node.get(try_pred);

				 Cancel if no significant difference between sets
				if (base_node.idx2oedges.containsKey(try_node.idx)) {
					float entRate = base_node.idx2oedges.get(try_node.idx).sim;
					System.out.println(ANSI_BLUE + entRate + " entails " + try_pred + ANSI_RESET);
					if (entRate > 0.98) {
						continue;
					}
				}
				Map<Node, Float> imperfective_entailments = getEntailments(try_node, graph); */

				// Find modified pred entries and keep only those that are not perfectly aligned with pred
				List<Node> imperfective_nodes = Stream.of("try__", "trying__", "tried__", "failed__")
						.map(x -> x + base_pred)
						.filter(graph.pred2node::containsKey)
						.map(graph.pred2node::get)
						.filter(n ->
								!base_node.idx2oedges.containsKey(n.idx) ||
								base_node.idx2oedges.get(n.idx).sim < 0.98)
						.collect(Collectors.toList());

				if (imperfective_nodes.isEmpty()) { continue; }

				System.out.println(ANSI_YELLOW + formatPred(base_pred) + ANSI_RESET);
				imperfective_nodes.forEach(n -> System.out.println(ANSI_BLUE + formatPred(n.id) + ANSI_RESET));

				// Generate entailment map with remaining entries
				Map<Node, Float> imperfective_entailments = imperfective_nodes.stream()
						.map(n -> getEntailments(n, graph, false))
						.flatMap(map -> map.entrySet().stream())
						.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, Math::max));

				// Generate base entailment map
				Map<Node, Float> perfective_entailments = getEntailments(base_node, graph, false);
				Map<Node, Float> intersection_entailments = new HashMap<>(perfective_entailments);

				// Do set arithmetic to divide into: (only base) (intersection) (only try)
				Set<Node> perfective_keys = perfective_entailments.keySet();
				Set<Node> imperfective_keys = imperfective_entailments.keySet();
				Set<Node> intersection_keys = intersection_entailments.keySet();
				intersection_keys.retainAll(imperfective_keys);

				intersection_keys.forEach(key -> intersection_entailments.put(key,
						perfective_entailments.get(key) - imperfective_entailments.get(key)));

				perfective_keys.removeAll(intersection_keys);
				imperfective_keys.removeAll(intersection_keys);

				// Sort and print each entailment set
				System.out.println("\n--{perfective}------");
				printEntailmentMapSorted(base_node, perfective_entailments);

				System.out.println("\n--{intersection}------");
				printEntailmentMapSorted(base_node, intersection_entailments);

				System.out.println("\n--{imperfective}------");
				printEntailmentMapSorted(base_node, imperfective_entailments);

				System.out.println("\n========================");
				System.out.println();
			}
		}
	}

//	public static Map<Node, Map<Node, Float>> getEntailmentsForModifierInGraph(String modifier, PGraph graph) {
//
//	}

	public static Map<String, Map<String, Map<Node, Map<Node, Float>>>> getEntailmentSetsForModifiers(Set<String> modifiers, PGraph graph, boolean weightConfidence) {
		Map<String, Map<String, Map<Node, Map<Node, Float>>>> entsets = new HashMap<>();
		for (String modifier : modifiers) {
			entsets.put(modifier, new HashMap<>());
		}

		for (String modifier : modifiers) {
			String mod = modifier + "__";
			Map<Node, Map<Node, Float>> ents = graph.nodes.stream()
					.filter(n -> n.id.contains(mod))
					.collect(Collectors.toMap(n -> n, n -> getEntailments(n, graph, weightConfidence)));

			entsets.get(modifier).put(graph.types, ents);
		}

		return entsets;
	}

	public static Map<String, Map<String, Float>> entNodesToEntPreds(Map<Node, Map<Node, Float>> entNodes, boolean stripModifiersAntecedent) {
		Map<String, Map<String, Float>> entPreds = new HashMap<>();
		for (Node n : entNodes.keySet()) {
			String antecedent = stripModifiersAntecedent ? n.id.substring(n.id.indexOf("(")) : n.id;
			Map<String, Float> entailments = entNodes.get(n).entrySet().stream().collect(Collectors.toMap(e -> e.getKey().id, Map.Entry::getValue));
			entPreds.put(antecedent, entailments);
		}
		return entPreds;
	}

	public static Float entsetDivergance(Map<String, Map<String, Float>> base, Map<String, Map<String, Float>> comp) {
		float tp = 0;
		float fp = 0;
		float fn = 0;

		for (String key : base.keySet()) {
			if (!comp.containsKey(key)) { continue; }
			int intersectionSize = Sets.intersection(base.get(key).keySet(), comp.get(key).keySet()).size();
			tp += intersectionSize;
			fp += comp.get(key).size() - intersectionSize;
			fn += base.get(key).size() - intersectionSize;
		}

		float precision = tp / (tp + fp);
		float recall = tp / (tp + fn);
		float f1 = (2 * precision * recall) / (precision + recall);

		if (!Float.isNaN(f1)) {
			System.out.print(String.format("tp:%f\tfp:%f\tfn:%f\tp:%f\tr:%f\t", tp, fp, fn, precision, recall));
		}

		return f1;
	}

	public static int getConfidence(String id_ant, String id_con) {
		int antOccurrences = PGraph.predToOcc.get(id_ant);
		int conOccurrences = PGraph.predToOcc.get(id_con);
		return min(antOccurrences, conOccurrences);
	}

	public static Map<Node, Float> getEntailments(Node node, PGraph graph, boolean weightConfidence) {
		Map<Node, Float> entailments = new HashMap<>();
		for (Oedge e : node.oedges) {
			if (e.sim < 0.15) { continue; }
			Node entailedNode = graph.idx2node.get(e.nIdx);
			float score = e.sim;
			if (weightConfidence) {
				int confidence = getConfidence(node.id, entailedNode.id);
				if (confidence < 3) { continue; }
				score *= confidence;
			}
			entailments.put(entailedNode, score);
		}
		return entailments;
	}

	public static void printEntailmentMapSorted(Node node, Map<Node, Float> map) {
		printEntailmentMapSorted(node, map, -1);
	}

	public static void printEntailmentMapSorted(Node node, Map<Node, Float> map, int limit) {
		Stream<Map.Entry<Node, Float>> sorted = map.entrySet().stream()
				.sorted(Collections.reverseOrder(Map.Entry.comparingByValue()));
		if (limit >= 0) {
			sorted = sorted.limit(limit);
		}
		sorted.forEach(entry -> System.out.println(
				round.format(entry.getValue())
						+ " (" + getConfidence(node.id, entry.getKey().id) + ") " + formatPred(entry.getKey().id)));
	}

	public static void printPredicateModifiers() {
		Multiset<String> modifiers = TreeMultiset.create();
		Pattern pattern = Pattern.compile("(.*?)__");
		for (PGraph pgraph : GraphSet.generator()) {
			if (pgraph.nodes.size() == 0) {
				continue;
			}
			System.out.println("Reading in " + pgraph.types);
			for (Node n : pgraph.nodes) {
				String pred = n.id;
				if (!pred.contains("__")) {
					continue;
				}
				Matcher matcher = pattern.matcher(pred);
				while (matcher.find()) {
					String modifier = matcher.group(1);
					modifiers.add(modifier);
				}
			}
		}
		System.out.println("Processing...");
		Multiset<String> sortedModifiers = Multisets.copyHighestCountFirst(modifiers);
		sortedModifiers.entrySet().stream().filter(x -> x.getCount() > 10).forEach(System.out::println);
	}

	public static void printPredicatesContaining(String elt) {
		for (PGraph pgraph : GraphSet.generator()) {
			pgraph.pred2node.keySet().stream().filter(x -> x.contains(elt)).forEach(System.out::println);
		}
	}

	public static void printModifierDeverganceFromFailure(String modifier, boolean weightConfidence) {
		System.out.println("== Divergance of \"" + modifier + "\" from failed");

		final String failString = "failed";

		Set<String> modifiers = Sets.newHashSet(failString, modifier);

		if (weightConfidence) {
			System.out.println("Reading occurrence files...");
			PGraph.setPredToOcc(ConstantsGraphs.root);
		}
		System.out.println("Scanning graphs...");

		for (PGraph graph : GraphSet.generator()) {
			if (graph.nodes.isEmpty()) { continue; }
			if (weightConfidence) {
				graph.setSortedEdges();
			}

			Map<String, Map<String, Map<Node, Map<Node, Float>>>> entNodes = getEntailmentSetsForModifiers(modifiers, graph, weightConfidence);
			Map<String, Map<Node, Map<Node, Float>>> entNodesFail = entNodes.get(failString);
			Map<String, Map<Node, Map<Node, Float>>> entNodesTry = entNodes.get(modifier);

			for (String type : entNodesTry.keySet()) {
				if (!entNodesFail.containsKey(type)) { continue; }
				Map<String, Map<String, Float>> entPredsFail = entNodesToEntPreds(entNodesFail.get(type), true);
				Map<String, Map<String, Float>> entPredsTry = entNodesToEntPreds(entNodesTry.get(type), true);

				Float div = entsetDivergance(entPredsFail, entPredsTry);
				if (Float.isNaN(div)) { continue; }
				System.out.println(type + ": " + div);
			}
		}
	}

	public static void main(String[] args) throws FileNotFoundException {
		System.out.println("PROGRAM START");
		ConstantsGraphs.edgeThreshold = -1;

//		printPredicateEntailments(10);
//		printCausalEntailments();
//		printPredicateModifiers();
//		printPredicatesContaining("trying__");

		printModifierDeverganceFromFailure("trying", false);



		System.out.println("END");
	}

	static String formatPred(String pred) {
		List<String> predChunks = new ArrayList<>(Arrays.asList(pred.split("[()#]|\\.\\d|,", 0)));
		predChunks.removeAll(Arrays.asList(""));
		String basePred = "";
		if (predChunks.get(0).contains("__")) {
			basePred += predChunks.get(0);
			predChunks.remove(0);
		}
		basePred += predChunks.get(1);
		String arg1 = predChunks.get(2);
		String arg2 = predChunks.get(3);
		return String.format("%s %s %s (%d)", arg1, basePred, arg2, PGraph.predToOcc.get(pred));
	}

}
