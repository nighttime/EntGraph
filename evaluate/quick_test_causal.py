import argparse
import os

import utils
from proposition import Prop
from article import *
from entailment import *
from reference import *

from typing import *


def assess_graphs(graphs: EGraphCache):
    seen_graphs = set()
    for k,g in graphs.items():

        # t1, t2 = next(n for n in g.nodes if n.count('#') == 2).split('#')[1:]
        t1, t2 = k.split('#')

        first_types = ['person', 'location', 'organization', 'thing']
        second_types = ['organization', 'thing']

        if t1 not in first_types or t2 not in second_types:
            continue

        # if '_1' in t2:
        #     continue

        if g in seen_graphs:
            continue

        seen_graphs.add(g)

        triples = [
                ('(buy.1,buy.2)', '(own.1,own.2)', EdgeType.CONSEQUENCE_SUCCESS),
                ('(buy.1,buy.2)', '(possess.1,possess.2)', EdgeType.CONSEQUENCE_SUCCESS),
                ('(buy.1,buy.2)', '(obtain.1,obtain.2)', EdgeType.CONSEQUENCE_SUCCESS),
                ('(acquire.1,acquire.2)', '(own.1,own.2)', EdgeType.CONSEQUENCE_SUCCESS),
                ('(acquire.1,acquire.2)', '(possess.1,possess.2)', EdgeType.CONSEQUENCE_SUCCESS),
                ('(acquire.1,acquire.2)', '(obtain.1,obtain.2)', EdgeType.CONSEQUENCE_SUCCESS),

                ('(sell.1,sell.2)', '(own.1,own.2)', EdgeType.PRECONDITION),
                ('(sell.1,sell.2)', '(possess.1,possess.2)', EdgeType.PRECONDITION)
        ]

        if t1 != t2:
            forw_typing = '#' + t1 + '#' + t2
        else:
            forw_typing = '#' + t1 + '_1#' + t2 + '_2'
        typed_triples = [(r1+forw_typing, r2+forw_typing, e) for r1,r2,e in triples]
        # rels = {r for r1,r2,_ in typed_triples for r in [r1,r2]}
        # if not all(r in g.nodes for r in rels):
        #     continue

        connector = {EdgeType.CONSEQUENCE_SUCCESS: '->', EdgeType.PRECONDITION: '>-'}
        opposite = {EdgeType.CONSEQUENCE_SUCCESS: EdgeType.PRECONDITION, EdgeType.PRECONDITION: EdgeType.CONSEQUENCE_SUCCESS}

        print(forw_typing)
        for t in typed_triples:
            if t[0] not in g.nodes:
                print(tcolors.WARNING + '?   ' + tcolors.ENDC + '\t{}'.format(t[0]))
                continue
            elif t[1] not in g.nodes:
                print(tcolors.WARNING + '?   ' + tcolors.ENDC + '\t{}'.format(t[1]))
                continue

            res = [e.score for e in g.get_entailments(t[0]) if e.pred == t[1] and e.edge_type == t[2]]
            if res:
                print(tcolors.OKGREEN + 'PASS' + tcolors.ENDC + '\t{} {} {}\t: {}'.format(t[0], connector[t[2]], t[1], res[0]))
            else:
                res_opp = [e.score for e in g.get_entailments(t[0]) if e.pred == t[1] and e.edge_type == opposite[t[2]]]
                if res_opp:
                    print(tcolors.FAIL + 'FAIL' + tcolors.ENDC + '\t{} {} {}'.format(t[0], connector[t[2]], t[1]))
                else:
                    print(tcolors.OKGREEN + '----' + tcolors.ENDC + '\t{} {} {}'.format(t[0], connector[t[2]], t[1]))
        utils.print_bar()




def main():
    global ARGS
    ARGS = parser.parse_args()

    print('Reading model files from {} ...'.format(ARGS.graphs))
    if ARGS.text_graphs:
        graphs = read_graphs(ARGS.graphs, EGStage.LOCAL, keep_forward=True)
    else:
        graphs = read_precomputed_EGs(ARGS.graphs)

    assert list(graphs.values())[0].typed_edges == True
    assess_graphs(graphs)


parser = argparse.ArgumentParser(description='Yago compatibility with News Data')
parser.add_argument('graphs', help='Path to causal graph zip for use in question answering')
parser.add_argument('--text-graphs', action='store_true', help='Flag if graphs are to be read in from raw text files')

if __name__ == '__main__':
    main()