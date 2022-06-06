# This file is to be used for common reference across modules
from enum import Enum

try:
	RUNNING_LOCAL = bool(int(open('RUNNING_LOCAL').readline()))
except:
	RUNNING_LOCAL = bool(int(open('../RUNNING_LOCAL').readline()))

FINISH_TIME = None

DS_SHORT_NAME_TO_DISPLAY = {'levy_holt': 'Levy/Holt', 'sl_ant': 'ANT'}

# EVAL HYPERPARAMETERS
class SMOOTH(Enum):
	ALL 	= 'ALL'
	MISSING = 'MISSING'
	NONE 	= 'NONE'

class TRIGGER(Enum):
	NO_EG_SUPPORT	 = 'NO_EG_SUPPORT'		# Smooth if EG finds no support, combine with EG
	NO_EG_UNCOMBINED = 'NO_EG_UNCOMBINED' 	# Smooth if EG finds no support, DO NOT combine with EG
	ALWAYS			 = 'ALWAYS'				# Always smooth

class WN_RELATION(Enum):
	SYNONYM 	= 'synonyms'
	ANTONYM		= 'antonyms'
	HYPERNYM 	= 'hypernyms'
	HYPONYM		= 'troponyms'

SMOOTHING_K = 4
SMOOTH_P = SMOOTH.MISSING
SMOOTH_Q = SMOOTH.MISSING
SMOOTHING_SELF_LOOPS = True
SMOOTHING_TRIGGER = TRIGGER.NO_EG_SUPPORT
P_WN_RELATION = WN_RELATION.HYPERNYM
Q_WN_RELATION = WN_RELATION.HYPONYM


GRAPH_BACKOFF = None


K_UNARY_ENT_MENTIONS = 6
K_BINARY_ENT_MENTIONS = 6

K_UNARY_PRED_MENTIONS = 5
K_BINARY_PRED_MENTIONS = 5



MODAL_VERBS = {'can', 'may', 'must', 'could', 'would', 'should', 'might', 'shall', 'will', 'ought'}
AUXILIARY_VERBS = {'do', 'have'} | MODAL_VERBS # also 'be' but we like the linking verb!
LIGHT_VERBS = {'take', 'make'}
REPORTING_VERBS = {'say'}
PREPOSITIONS = {
'about',
'beside',
'near',
'to',
'above',
'between',
'of',
'towards',
'across',
'beyond',
'off',
'under',
'after',
'by',
'on',
'underneath',
'against',
'despite',
'onto',
'unlike',
'along',
'down',
'opposite',
'until',
'among',
'during',
'out',
'up',
'around',
'except',
'outside',
'upon',
'as',
'for',
'over',
'via',
'at',
'from',
'past',
'with',
'before',
'in',
'round',
'within',
'behind',
'inside',
'since',
'without',
'below',
'into',
'than',
'beneath',
'like',
'through'
}


class tcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'