import random
import csv


def g_0(n):
    return("?", )*n

def s_0(n):
    return('∅', )*n

def more_general(h1, h2):
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == "?"or(x != "∅"and(x == y or y == '∅'))
        more_general_parts.append(mg)
    return all(more_general_parts)


def fulfills(example, hypothesis):
    return more_general(hypothesis, example)


def min_generalizations(h, x):
    h_new = list(h)
    for i in range(len(h)):
        if not fulfills(x[i:i+1], h[i:i+1]):
            h_new[i] = '?' if h[i] != '∅'else x[i]
    return [tuple(h_new)]


def min_specializations(h, domains, x):
    results = []
    for i in range(len(h)):
        if h[i] == "?":
            for val in domains[i]:
                if x[i] != val:
                    h_new = h[:i]+(val, )+h[i+1:]
                    results.append(h_new)
                elif h[i] != "∅":
                    h_new = h[:i]+('∅', )+h[i+1:]
                    results.append(h_new)
    return results


with open('Sunny.csv')as csvFile:
    examples = [tuple(line)for line in csv.reader(csvFile)]


def get_domains(examples):
    d = [set()for i in examples[0]]
    for x in examples:
        for i, xi in enumerate(x):
            d[i].add(xi)
    return[list(sorted(x))for x in d]
get_domains(examples)


def candidate_elimination(examples):
    domains = get_domains(examples)[:-1]
    G = set([g_0(len(domains))])
    S = set([s_0(len(domains))])
    i = 0
    print("\nG[{0}]:".format(i), G)
    print("\nS[{0}]:".format(i), S)
    for xcx in examples:
        i = i+1
        x, cx = xcx[:-1], xcx[-1]
        if cx == 'YES':
            G = {g for g in G if fulfills(x, g)}
            S = generalize_s(x, G, S)
        else:
            S = {s for s in S if not fulfills(x, s)}
            G = specialize_g(x, domains, G, S)
        print("\n G[{0}]:".format(i), G)
        print("\nS[{0}]:".format(i), S)
    return


def generalize_s(x, G, S):
    S_prev = list(S)
    for s in S_prev:
        if s not in S:
            continue
        if not fulfills(x, s):
            S.remove(s)
            Splus = min_generalizations(s, x)
            S.update([h for h in Splus if any([more_general(g, h)for g in G])])
            S.difference_update([h for h in S if any([more_general(h, h1) for h1 in S if h != h1])])
    return S


def specialize_g(x, domains, G, S):
    G_prev = list(G)
    for g in G_prev:
        if g not in G:
            continue
        if fulfills(x, g):
            G.remove(g)
            Gminus = min_specializations(g, domains, x)
            G.update([h for h in Gminus if any([more_general(h, s) for s in S])])
            G.difference_update([h for h in G if any([more_general(g1, h)for g1 in G if h != g1])])
    return G


candidate_elimination(examples)