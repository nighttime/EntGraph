# This file is to be used for common reference across modules

# RUNNING_LOCAL = True
try:
	RUNNING_LOCAL = bool(int(open('RUNNING_LOCAL').readline()))
except:
	RUNNING_LOCAL = bool(int(open('../RUNNING_LOCAL').readline()))

K_UNARY_ENT_MENTIONS = 5
K_BINARY_ENT_MENTIONS = 5

K_UNARY_PRED_MENTIONS = 3
K_BINARY_PRED_MENTIONS = 3

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

