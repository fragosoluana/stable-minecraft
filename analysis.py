import sys
import collections
import community
import jellyfish
import os
import time
import psutil

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy import stats

import stable
import importlib
importlib.reload(stable)

##################
# --- Config --- #
##################
debug_path = ""
# debug_path = ""
f = open(debug_path + "info", "r").readlines()
user = f[0].rstrip()
pwd = f[1].rstrip()
server = f[2].rstrip()
port = f[3].rstrip()
minecraft_db = f[4].rstrip()
engine = create_engine("mysql+pymysql://" + user + ":" + pwd + "@" + server + ":" + port + "/" + minecraft_db)

# plt.style.use('ggplot')
print("Python version: " + sys.version)


def pre_process():
    def get_xp():
        xp = []
        xp_level = []
        min_dimens = []
        for player in P["comp"]:
            max_xp = log_sims.loc[log_sims["comp"] == player, "xp"].max()
            xp_level.append(max_xp)

            if max_xp >= 21:
                xp.append("Expert")
            else:
                xp.append("Novice")

            min_dimens.append(log_sims.loc[log_sims["comp"] == player, "level"].min())

        return xp, xp_level, min_dimens

    def to_dc(df, interval=1):
        # Random start time to define the time groups
        dc_start = pd.to_datetime('2000-01-01 00:00:00')

        df = df.sort_values(by='time')
        df['time'] = pd.to_datetime(df['time'])

        # Create groups for the same duty cycle (e.g, each 5mins)
        df['dc'] = ((df['time'] - dc_start).dt.total_seconds() / interval).astype(int)

        return df

    print("Downloading minecraft dataset")
    minecraft = pd.read_sql("SELECT time, player_name as comp, position_x as pos_x, position_y as pos_z, "
                        "position_z as pos_y, experience_level as xp, dimension as level " \
                        "FROM player_data_filtered", engine)

    print("Downloading motivation dataset")
    sims = pd.read_sql("SELECT uid as comp, sims_motivation as mot, sims_id_regulation as id_reg, "
                       "sims_ex_regulation as ex_reg, sims_amotivation as amot "
                       "FROM postsurvey_calc", engine)
    sims["comp"] = sims["comp"].astype(str)

    log_sims = minecraft[minecraft["comp"].isin(sims["comp"])]
    print("# rows after SIMS filtering: ", len(log_sims))

    P = pd.DataFrame({"comp": list(log_sims["comp"].unique())})
    P["xp"], P["xp_level"], P["dimen"] = get_xp()
    P = P.merge(sims, how="left", on="comp")
    P = P.sort_values(by=["xp_level", "comp"], ascending=False).reset_index(drop=True)
    print("Xp Class Distribution:", collections.Counter(P['xp']))

    print("Downloading action dataset")
    sql = "SELECT time, player_name as comp FROM player_interaction where player_name IN %s"
    sql = sql.replace("%s", str(tuple(P['comp'].astype(int).unique())))
    action = pd.read_sql(sql, engine)

    log_sims = to_dc(log_sims)
    action = to_dc(action)

    print("Aggregating and merging datasets")
    log_sims = log_sims.groupby(["comp", "dc"], as_index=False).agg({"pos_x": np.mean, "pos_y": np.mean,
                                                         "pos_z": np.mean, "level": 'last'})
    action = action.groupby(['comp', 'dc'], as_index=False).size().reset_index(name="n_activity")

    G = log_sims.merge(action, on=["comp", "dc"], how="left").fillna(0)

    return G, P


def plot_boxplot(players_sim, players, analysis, c_name, min_ngram, max_ngram=0, max_y=1.0, xp=True, mot=None):
    if mot is None:
        fig = plt.figure(figsize=(13, 6))
    else:
        fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    if max_ngram == 0:
        max_ngram = min_ngram

    if mot is None:
        if min_ngram == max_ngram:
            plt.title("Similarities " + analysis + "-player analysis with N = " + str(min_ngram), fontsize=17)
        else:
            plt.title("Similarities " + analysis + "-player analysis with min N = " + str(min_ngram) + " and max N = "
                      + str(max_ngram), fontsize=17)
    else:
        plt.title("Intrinsic motivation scores", fontsize=17)
    plt.ylim(0, max_y)

    if mot is None:
        bp = ax.boxplot(players_sim, positions=players.index, patch_artist=True)
    else:
        mot_scores = pd.DataFrame({"score": players_sim, "group": list(mot.sort_values(["comp"])["partition"])})
        mots = [mot_scores.loc[mot_scores["group"] == 0, "score"], mot_scores.loc[mot_scores["group"] == 1, "score"]]
        bp = ax.boxplot(mots, patch_artist=True)

    # Customize the axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    if mot is None:
        n_boxes = len(players)
    else:
        n_boxes = len(mots)

    ## change color depending on the competence
    for i in range(0, n_boxes):
        if xp:
            if players.loc[i, 'xp'] == 'Expert':
                # change outline and fill box color
                bp['boxes'][i].set(facecolor='#006ba4', linewidth=0.1)

            if players.loc[i, 'xp'] == 'Novice':
                bp['boxes'][i].set(facecolor='#a2c8ec', linewidth=0.1)
        elif mot is not None:
            if i == 0:
                bp['boxes'][i].set(facecolor='#ff9e4a', linewidth=0.1)
            else:
                bp['boxes'][i].set(facecolor='#ed665d', linewidth=0.1)
        else:
            bp['boxes'][i].set(facecolor='#ababab', linewidth=0.1)

        bp['whiskers'][i * 2].set(linewidth=0.5)
        bp['whiskers'][i * 2 + 1].set(linewidth=0.5)
        bp['caps'][i * 2].set(linewidth=0.5)
        bp['caps'][i * 2 + 1].set(linewidth=0.5)
        bp['medians'][i].set(linewidth=1.5, color='#ff800e')
        bp['fliers'][i].set(marker='o', color='#898989', alpha=0.5, markersize=3, markerfacecolor='#898989',
                           markeredgecolor='#898989')

    if mot is None:
        plt.ylabel("Similarity Score", fontsize=17)
        plt.xlabel("Player ID", fontsize=17)
        plt.savefig(debug_path + 'results/' + str(c_name) + '/sim_boxplot_' + analysis + '_ngram' + str(min_ngram) + '.png', dpi=500)
    else:
        plt.ylabel("Scores (%)", fontsize=17)
        plt.xlabel("Partitions", fontsize=17)
        plt.savefig(debug_path + 'results/' + str(c_name) + '/mot_scores.png',dpi=500)

    plt.close()


def plot_bar(y, players, c_name, min_ngram=None, max_ngram=None, analysis='', sim=True, xp=True, mot=None):
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    plt.xticks(players.index)
    plt.xlabel("Player ID", fontsize=17)

    if max_ngram is None:
        max_ngram = min_ngram

    color_map = []

    if xp:
        for comp in players['xp']:
            if comp == 'Expert':
                color_map.append('#006ba4')
            if comp == 'Novice':
                color_map.append('#a2c8ec')
    elif mot is not None:
        for comp in players.index:
            if mot.loc[mot['comp'] == comp, 'partition'].values[0] == 0:
                color_map.append('#ff9e4a')
            if mot.loc[mot['comp'] == comp, 'partition'].values[0] == 1:
                color_map.append('#ed665d')
    else:
        color_map = '#ababab'

    ax.bar(players.index, y, color=color_map, edgecolor='black', linewidth=0.1)

    # Customize the axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    if sim:
        if min_ngram == max_ngram:
            plt.title("Similarities " + analysis + "-player analysis with N = " + str(min_ngram), fontsize=17,
                      label=["Advanced", "Non-Advanced"])
        else:
            plt.title("Similarities " + analysis + "-player analysis with min N = " + str(min_ngram) + " and max N = "
                      + str(max_ngram), fontsize=17)
        plt.ylabel("Similarity Score", fontsize=17)
        plt.ylim(0, 1)
        plt.savefig(debug_path + 'results/' + str(c_name) + '/sim_bar_' + analysis + '_ngram' + str(min_ngram) + '.png', dpi=500)
    elif mot is not None:
        plt.ylabel("Scores (%)", fontsize=17)
        plt.title("Intrinsic motivation scores", fontsize=17)
        plt.savefig(debug_path + 'results/' + str(c_name) + '/mot_scores_bar.png', dpi=500)
    else:
        plt.ylabel("Word Count", fontsize=17)
        plt.title("Word count per player", fontsize=17)
        plt.savefig(debug_path + 'results/' + str(c_name) + '/word_len.png', dpi=500)
    plt.close()


def plot_graph(G, labels, threshold, c_name, ngram, xp=True):
    pos = nx.kamada_kawai_layout(G)

    color_map = []
    for node in G.nodes.keys():
        if xp:
            if labels[node] == 'Expert':
                color_map.append('#006ba4')
            if labels[node] == 'Novice':
                color_map.append('#a2c8ec')
        else:
            if labels.loc[labels['comp'] == node, 'partition'].values[0] == 0:
                color_map.append('#ff9e4a')
            if labels.loc[labels['comp'] == node, 'partition'].values[0] == 1:
                color_map.append('#ed665d')
            if labels.loc[labels['comp'] == node, 'partition'].values[0] == 2:
                color_map.append('#ad8bc9')
            if labels.loc[labels['comp'] == node, 'partition'].values[0] == 3:
                color_map.append('#a8786e')

    # edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    plt.figure()
    plt.title("Similarities greater than " + str(int(threshold * 100)) + "% with N = " + str(ngram))
    nx.draw(G, pos, with_labels=True, node_color=color_map, edge_color="silver",
            font_size=11, width=0.5, font_color="black")

    plt.savefig(debug_path + 'results/' + str(c_name) + "/sim" + str(threshold) + "_ngram" + str(ngram) + '.png', dpi=500)
    plt.close()


def louvain(sims, participants, c_name, threshold, ngram):
    similars = stable.filter_sims(sims, participants, threshold)
    similars = sorted(similars, key=lambda x: x[2])

    if len(similars) > 0:
        G = nx.Graph()
        G.add_weighted_edges_from(similars)
        w_degrees = sorted(G.degree(weight='weight'), key=lambda x: x[0])
        print("w_degrees:", w_degrees)
        degrees = sorted(G.degree, key=lambda x: x[0])
        print("DEGREES:", degrees)
        degrees = [(w_degrees[i][0], w_degrees[i][1]/degrees[i][1]) for i in range(len(w_degrees))]
        degrees = sorted(degrees, key=lambda x: x[1])
        print("DEGREES AVG:", degrees)

        partitions = {}
        for i in range(len(degrees)-1, -1, -1):
            if i <= (len(degrees)//2):
                partitions[degrees[i][0]] = 0
            else:
                partitions[degrees[i][0]] = 1
        best_partition = community.best_partition(G, randomize=False, partition=partitions)
        partitions = []
        for i in best_partition:
            partitions.append((i, best_partition[i]))
        partitions = pd.DataFrame(partitions, columns=['comp', 'partition'])

        plot_graph(G, partitions, threshold, c_name, ngram, False)

        return partitions


def stat_test(sims, players, btw=True):
    if btw:
        between = []
        for i in players.index:
            for j in sims[i]:
                between.append({'comp': i, 'score': j, 'xp': players.loc[i, 'xp']})

        between = pd.DataFrame(between).groupby(['comp'], as_index=False).agg({'xp': 'first', 'score': 'mean'}) \
            .sort_values(['comp'])

        stat, p = stats.mannwhitneyu(between.loc[0:18, 'score'], between.loc[18:30, 'score'])
        return p
    else:
        # Check if sims within is a normal distribution. If it is not a normal distribution, the Mann-Whiteny-U test is
        # applied with the null hypothesis that the Advanced and Beginner players are from the same population.
        withins = []
        for i in range(0, len(sims)):
            comp = 'Advanced'
            if i > 17:
                comp = 'Beginner'
            withins.append((comp, sims[i]))

        stat, p = stats.mannwhitneyu(sims[0:18], sims[18:30])
        return p


def mot_stat_test(thresholds, partitions, players):
    sigfs = []

    for i in players.columns:
        if i not in ["comp", "xp", "xp_level"]:
            mot = i

            for threshold in thresholds:
                merged = partitions.merge(players[['comp', mot]], how='left', left_on='comp', right_index=True)
                merged.columns = ['new_comp', 'partition', 'comp', mot]

                n_part = max(merged['partition'])

                if n_part == 0:
                    continue

                if n_part == 1:
                    p0 = merged.loc[merged['partition'] == 0, [mot]].values
                    p1 = merged.loc[merged['partition'] == 1, [mot]].values

                    F, p = stats.mannwhitneyu(p0, p1)

                elif n_part == 2:
                    p0 = merged.loc[merged['partition'] == 0, [mot]].values
                    p1 = merged.loc[merged['partition'] == 1, [mot]].values
                    p2 = merged.loc[merged['partition'] == 2, [mot]].values

                    F, p = stats.kruskal(p0, p1, p2)

                elif n_part == 3:
                    p0 = merged.loc[merged['partition'] == 0, [mot]].values
                    p1 = merged.loc[merged['partition'] == 1, [mot]].values
                    p2 = merged.loc[merged['partition'] == 2, [mot]].values
                    p3 = merged.loc[merged['partition'] == 3, [mot]].values

                    F, p = stats.kruskal(p0, p1, p2, p3)

                if p < 0.05:
                    sigfs.append({"sim": threshold, "n_p": len(merged["new_comp"]),
                              "clusters": n_part + 1, "p_value": p, "mot": mot})

    return sigfs

# ------------------
# StABLE
# ------------------
G, P = pre_process()
sims_loc = stable.run(G, P, stable.Construction.loc)
sims_dt = stable.run(G, P, stable.Construction.dwell_trip)

sims_dta = stable.run(G, P, stable.Construction.dwell_trip_action)
# ------------------
# PLOTS
# ------------------
#RQ1, RQ2
c_name = stable.Construction.loc
min_ngram = c_name.value

print("BTW Range-N highest similarity:", sims_loc["btw"]["range_n"][sims_loc["btw"]["range_n"] < 0.99999].max())
print("BTW Range-N similarity avg:", sims_loc["btw"]["range_n"][sims_loc["btw"]["range_n"] < 0.99999].mean())
print("BTW Range-N similarity std:", sims_loc["btw"]["range_n"][sims_loc["btw"]["range_n"] < 0.99999].std())
print("LOC wth p value:", stat_test(sims_loc["btw"]["range_n"], P))

print("BTW Max-N highest similarity:", sims_loc["btw"]["max_n"][sims_loc["btw"]["max_n"] < 0.99999].max())
print("BTW Max-N similarity avg:", sims_loc["btw"]["max_n"][sims_loc["btw"]["max_n"] < 0.99999].mean())
print("BTW Max-N similarity std:", sims_loc["btw"]["max_n"][sims_loc["btw"]["max_n"] < 0.99999].std())
print("LOC wth p value:", stat_test(sims_loc["btw"]["max_n"], P))

print("BTW Min-N highest similarity:", sims_loc["btw"]["min_n"][sims_loc["btw"]["min_n"] < 0.99999].max())
print("BTW Min-N similarity avg:", sims_loc["btw"]["min_n"][sims_loc["btw"]["min_n"] < 0.99999].mean())
print("BTW Min-N greater than 0.5:", len(sims_loc["btw"]["min_n"][sims_loc["btw"]["min_n"] < 0.99999])/2 -
      len(sims_loc["btw"]["min_n"][sims_loc["btw"]["min_n"] < 0.5])/2)
plot_boxplot(sims_loc["btw"]["min_n"], P, "between", c_name, min_ngram, max_y=1)
print("LOC btw p value:", stat_test(sims_loc["btw"]["min_n"], P))


print("WTH Min-N highest similarity:", max(sims_loc["wth"]["min_n"]))
print("WTH Min-N similarity avg:", np.mean(sims_loc["wth"]["min_n"]))
print("WTH Min-N similarity std:", np.std(sims_loc["wth"]["min_n"]))
plot_bar(sims_loc["wth"]["min_n"], P, c_name, min_ngram, analysis="within")
print("LOC wth p value:", stat_test(sims_loc["wth"]["min_n"], P, btw=False))

print("WTH Range-N highest similarity:", max(sims_loc["wth"]["range_n"]))
print("WTH Range-N similarity avg:", np.mean(sims_loc["wth"]["range_n"]))
print("WTH Range-N similarity std:", np.std(sims_loc["wth"]["range_n"]))
print("LOC wth p value:", stat_test(sims_loc["wth"]["range_n"], P, btw=False))

print("WTH Max-N highest similarity:", max(sims_loc["wth"]["max_n"]))
print("WTH Max-N similarity avg:", np.mean(sims_loc["wth"]["max_n"]))
print("WTH Max-N similarity std:", np.std(sims_loc["wth"]["max_n"]))
print("LOC wth p value:", stat_test(sims_loc["wth"]["max_n"], P, btw=False))


#RQ3
c_name = stable.Construction.dwell_trip
min_ngram = c_name.value
plot_boxplot(sims_dt["btw"]["range_n"], P, "between", c_name, 1, 4)
plot_boxplot(sims_dt["btw"]["min_n"], P, "between", c_name, min_ngram)
plot_boxplot(sims_dt["btw"]["max_n"], P, "between", c_name, min_ngram * 2)
plot_bar([len(x.split()) for x in sims_dt["btw"]["feature"]], P, c_name, sim=False)

print("DT btw-min p value:", stat_test(sims_dt["btw"]["min_n"], P))
print("DT btw-range p value:", stat_test(sims_dt["btw"]["range_n"], P))
print("DT btw-max p value:", stat_test(sims_dt["btw"]["max_n"], P))

threshold = 0.6
partitions = louvain(sims_dt["btw"]["min_n"], P, c_name, threshold, min_ngram)
print(mot_stat_test([threshold], partitions, P))

G_sims = stable.to_graph(sims_dt["btw"]["min_n"], P, 0.5)
plot_graph(G_sims, P["xp"], 0.5, c_name, min_ngram)
print("DEGREES:", sorted(G_sims.degree, key=lambda x: x[1]))

print("DT wth-min p value:", stat_test(sims_dt["wth"]["min_n"], P, btw=False))
print("DT wth-range p value:", stat_test(sims_dt["wth"]["range_n"], P, btw=False))
print("DT wth-max p value:", stat_test(sims_dt["wth"]["max_n"], P, btw=False))

#RQ4
c_name = stable.Construction.dwell_trip_action
min_ngram = c_name.value
plot_boxplot(sims_dta["btw"]["range_n"], P, "between", c_name, 1, 8)
print("DTA btw-range p value:", stat_test(sims_dta["btw"]["range_n"], P))

plot_boxplot(sims_dta["btw"]["min_n"], P, "between", c_name, min_ngram)
print("DTA btw-min p value:", stat_test(sims_dta["btw"]["min_n"], P))

plot_boxplot(sims_dta["btw"]["max_n"], P, "between", c_name, min_ngram * 2)
print("DTA btw-max p value:", stat_test(sims_dta["btw"]["max_n"], P))

plot_bar([len(x.split()) for x in sims_dta["btw"]["feature"]], P, c_name, sim=False)
plot_graph(stable.to_graph(sims_dta["btw"]["min_n"], P, 0.1), P["xp"], 0.1, c_name, min_ngram)

threshold = 0.1
partitions = louvain(sims_dta["btw"]["min_n"], P, c_name, threshold, min_ngram)
print(mot_stat_test([threshold], partitions, P))

plot_bar(P["mot"], P, c_name, sim=False, xp=False, mot=partitions)
plot_boxplot(P["mot"], P, "", c_name, 0, mot=partitions, xp=False)

#RQ5
c_name = stable.Construction.dwell_trip
min_ngram = c_name.value
plot_bar(sims_dt["wth"]["range_n"], P, c_name, 1, 4, analysis="within")
print("DT wth-range p value:", stat_test(sims_dt["wth"]["range_n"], P, btw=False))

plot_bar(sims_dt["wth"]["min_n"], P, c_name, min_ngram, analysis="within")
print("DT wth-min p value:", stat_test(sims_dt["wth"]["min_n"], P, btw=False))

plot_bar(sims_dt["wth"]["max_n"], P, c_name, min_ngram * 2, analysis="within")
print("DT wth-max p value:", stat_test(sims_dt["wth"]["max_n"], P, btw=False))

c_name = stable.Construction.dwell_trip_action
min_ngram = c_name.value
plot_bar(sims_dta["wth"]["range_n"], P, c_name, 1, 8, analysis="within players")
print("DTA wth-range p value:", stat_test(sims_dta["wth"]["range_n"], P, btw=False))

plot_bar(sims_dta["wth"]["min_n"], P, c_name, min_ngram, analysis="within players")
print("DTA wth-min p value:", stat_test(sims_dta["wth"]["min_n"], P, btw=False))

plot_bar(sims_dta["wth"]["max_n"], P, c_name, min_ngram * 2, analysis="within players")
print("DTA wth-max p value:", stat_test(sims_dta["wth"]["max_n"], P, btw=False))
