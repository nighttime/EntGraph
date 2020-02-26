package constants;

import entailment.vector.EntailGraphFactoryAggregator;

public class ConstantsAgg {

	// ####################################################################################################
	// Extract sims
	// All parameters:

	public static boolean onlyDSPreds = false;
	public static boolean rawExtractions = false;// gbooks original (or gbooks in general?)
	public static boolean GBooksCCG = false;
	public static boolean useTimeEx = false;
	public static boolean isCCG = true;
	public static boolean isTyped = true;
	public static boolean updatedTyping = true;// a few fixes here and there. flase for TACL experiments!
	public static boolean figerHierarchy = false;// for TACL experiments, we had it false
	public static boolean isForeign = false;
	public static boolean addTimeStampToFeats = false;
	public static final boolean normalizePredicate = true;// if rawExtraction, wouldn't matter.
	public static final boolean keepWillTense = false;// Must be false for normal entailment graphs
	public static final boolean backupToStanNER = true;
	public static final boolean onlyBinc = true;//vs all similarity measures. use to save memory and storage!
	
	public static boolean removeEventEventModifiers = false;
	public static boolean removeStopPreds = true;
	public static boolean removePronouns = false;//TODO: You must move this to parsing 
	public static boolean cutoffBasedonNSGraphs = true;// use NSpike-based cutoffs
	
	

	// cutoffs
	public static int minArgPairForPred = 10;// 100;
	public static int minPredForArgPair = 10;// 20;// min num of unique predicates for argpair
	// when NS based num aps, we allow x aps for each pred, even if not in NS
	public static int numArgPairsNSBasedAlwaysAllowed = 0;// default: 10
	public static int numTopTypePairs = 20;// the big types, used in NSbased sizes
	public static int maxPredsTotal = -1;// 35000;

	public static final int minPredForArg = -1;// min num of unique predicates for
	public static boolean removeGGFromTopPairs = true;// whether we should remove triples with two general entities
														// from top pairs

	public static final int numThreads = 10; //20 max?

	// embedding parameters
	public static boolean embBasedScores = false;// use sigmoid(transE score) instead of counts
	public static double sigmoidLocParameter = 10;
	public static double sigmoidScaleParameter = .25;
	public static int dsPredNumThreads = 15;
	public static EntailGraphFactoryAggregator.ProbModel probModel = EntailGraphFactoryAggregator.ProbModel.PEL;
	public static EntailGraphFactoryAggregator.LinkPredModel linkPredModel = EntailGraphFactoryAggregator.LinkPredModel.ConvE;


	public static String relAddress = "news_gen_argwise/news_gen_argwise.json";
//	public static String relAddress = "news_gen_argwise/news_gen_argwise_500k.json";
//	public static String relAddress = "/disk/scratch_big/jhosseini/mnt2/java/entGraph/news_genC.json";

	public static String NERAddress = "data/stan_NER/news_genC_stanNER.json";

//	public static String simsFolder = "newscrawl_sims/newscrawl_modifiers_" + minArgPairForPred + "_" + minPredForArgPair;
//	public static String simsFolder = "newsspike_sims/newsspike_argwise_" + relAddress.replaceAll("\\D","") + "k_" + minArgPairForPred + "_" + minPredForArgPair;
	public static String simsFolder = "newsspike_sims/newsspike_argwise_argnumbers_" + minArgPairForPred + "_" + minPredForArgPair;

	public static String foreinTypesAddress = "data/german_types.txt";// only important if isForeign=True

	public static boolean computeProbELSims = false;
	public static boolean linkPredBasedRandWalk = false;
	
	public static String NSSizesAddress;
	public static String NSPredSizesAddress;
	
	static {
		if (ConstantsAgg.addTimeStampToFeats) {
			NSSizesAddress = "data/NS_sizes_week_3_3.txt";
			NSPredSizesAddress = "data/NS_pred_sizes_week_3_3.txt";
		}
		else if (ConstantsAgg.figerHierarchy) {
			NSSizesAddress = "data/NS_sizes_hier_3_2.txt";
			NSPredSizesAddress = "data/NS_pred_sizes_hier_3_2.txt";
		}
		else {
			NSSizesAddress = "data/NS_sizes.txt";
			NSPredSizesAddress = "data/NS_pred_sizes.txt";
		}
	}

	// Generate graphs with entailments calculated for each predicate argument; also entailments with unaries
	public static boolean generateArgwiseGraphs = true;
	// Must have generateArgwiseGraphs = true in order to also make unary-only graphs
	public static boolean generateUnaryOnlyGraphs = true;
	// Include unary predicates which are possessive e.g. "Obama's strength"
	public static boolean keepPossessiveUnaries = false;
	// During graph building phase, build only 1/N graphs at a time to reduce memory constraints
	public static int numGraphBuildingPasses = 10;

}
