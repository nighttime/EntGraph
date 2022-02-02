import argparse
import os

import utils
from proposition import Prop
from article import *
from entailment import *
from reference import *

from typing import *

def scan_graph_file(fname, pleft, pright):
    with open(fname) as file:
        searching = False
        edge_type = None
        for line in file:
            if line.startswith('predicate: '):
                if searching:
                    print('Edge not found')
                    exit(0)
                pred = line.split()[1]
                if pred == pleft:
                    searching = True
            elif searching:
                if line.startswith('%'):
                    edge_type = EdgeType(line[1:].strip())
                    continue
                elif line.strip() == '' or line.startswith('num neighbors') or 'sims' in line:
                    continue
                else:
                    pred, score = line.split()
                    score = float(score)
                    if pred == pright:
                        print(tcolors.BOLD + '{} |= {} : {:.3f} ({})'.format(ARGS.left_pred, ARGS.right_pred, score, edge_type) + tcolors.ENDC)
                        exit(0)

        print('Node not found')


def main():
    global ARGS
    ARGS = parser.parse_args()

    print('Reading graph file: {} ...'.format(ARGS.graph))
    scan_graph_file(ARGS.graph, ARGS.left_pred, ARGS.right_pred)

    # graph = EntailmentGraph(ARGS.graph, keep_forward=True)
    # entailments = graph.get_entailments(ARGS.left_pred)
    # for e in entailments:
    #     if e.pred == ARGS.right_pred:
    #         print('{} |= {} : {:.3f} ({})'.format(ARGS.left_pred, ARGS.right_pred, e.score, e.edge_type))


parser = argparse.ArgumentParser(description='Check entailment graph for a score')
parser.add_argument('graph', help='Path to entailment graph text file')
parser.add_argument('left_pred', help='left-hand predicate for lookup')
parser.add_argument('right_pred', help='right-hand predicate for lookup')
# parser.add_argument('--text-graphs', action='store_true', help='Flag if graphs are to be read in from raw text files')

if __name__ == '__main__':
    main()