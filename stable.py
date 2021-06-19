import enum
import os
import time
import psutil

import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer


class Construction(enum.Enum):
    loc = 1
    dwell_trip = 2
    dwell_trip_action = 4


def run(df, P, c_name, cell_size=1, n_trip=3):
    df = df.copy(deep=True)
    df = df.sort_values(["dc", "comp"]).drop(columns=['dc'])

    if c_name != Construction.dwell_trip_action and "n_activity" in df.columns:
        df = df.drop(columns=['n_activity'])

    if "bin" not in df.columns:
        print("Stratifying")
        df = stratify(df, cell_size)

    print("Extracting features")
    f_btw, f_wth = extract_features(df, P, c_name, n_trip)

    print("Calculating similarities:")
    ngram = c_name.value

    process = psutil.Process(os.getpid())
    start_time = time.time()
    print("Min and max ngram:", ngram)
    n = calc_sims(f_btw, ngram, ngram, f_wth[0], f_wth[1])
    print("Min and max ngram:", ngram * 2)
    n2 = calc_sims(f_btw, ngram * 2, ngram * 2, f_wth[0], f_wth[1])
    print("Min ngram:", 1, " and Max ngram:", ngram*2)
    n2n = calc_sims(f_btw, 1, ngram * 2, f_wth[0], f_wth[1])
    print("--- STABLE --- Runtime (min): " + str((time.time() - start_time) / 60))
    print("--- STABLE --- Memory usage (GB): " + str(round(((process.memory_info().rss) / (10 ** 9)), 2)))

    return {"btw": {"feature": f_btw, "min_n": n[0], "max_n": n2[0], "range_n": n2n[0]},
            "wth": { "feature_before": f_wth[0], "feature_after": f_wth[1], "min_n": n[1], "max_n": n2[1],
                    "range_n": n2n[1]},
            "comp": P["comp"]}


def stratify(df, size):
    '''
    Stratifies the location points into a grid of cell size m
    :param x: column with the east-west location values
    :param y: column with the south-north location values
    :param size: integer representing the cell size in meters
    :return: Series with the grid values
    '''

    x = df["pos_x"].astype(float)
    y = df["pos_y"].astype(float)

    start_x = min(x)
    start_y = min(y)

    # Find the grids that each position point belongs to
    cell_x = np.ceil((x - start_x) / size).astype(int)
    cell_y = np.ceil((y - start_y) / size).astype(int)

    if "level" in df.columns:
        df["cell"] = df["level"].map(str) + "-"

    df["cell"] += cell_x.map(str) + "-" + cell_y.map(str)
    df = df.drop(columns=['level', 'pos_x', 'pos_y'])

    if "pos_z" in df.columns:
        z = df["pos_z"]
        start_z = min(z)
        cell_z = np.ceil((z - start_z) / size).astype(int)
        df["cell"] += "-" + cell_z.map(str)
        df = df.drop(columns=['pos_z'])

    return df


def extract_features(df, comps, c_name, n):
    def trip_detection(df):
        '''
        Operationalize trip. The trip is defined as changing locations for at least three consecutive downsampled
        time steps.

        :param df: data frame with the location cells
        :param n: integer for N-times duty cycle
        :return: dataframe with the length and duration of the trips
        '''

        df['dwell'] = (df['cell'] != df['cell'].shift()).cumsum()

        if c_name != Construction.dwell_trip_action:
            trip = df.groupby(['comp', 'dwell']).size().reset_index(name="duration")
            trip = trip.drop(columns=['dwell'])

        else:
            trip = df.groupby(['comp', 'dwell'], as_index=False).agg({"n_activity": ["sum", "count"]})
            trip.columns = ['comp', 'dwell', 'n_activity', 'duration']
            trip = trip.drop(columns=['dwell'])

        return trip

    def location_construction(p_movement):
        story = ""
        for cell in p_movement["cell"]:
            story += str(cell) + " "

        return story.rstrip()

    def dwell_trip_construction(p_movement, activity= False):
        story = ""
        trip_dur = 0
        for index, row in p_movement.iterrows():
            if row["duration"] > n:
                if trip_dur > 0:
                    story += "t" + str(trip_dur + 1) + " "
                    if activity:
                        story = add_action(story, row["n_activity"])
                    trip_dur = 0
                elif trip_dur == 0 and index > 0:
                    story += "t1 "
                    if activity:
                        story = add_action(story, row["n_activity"])

                story += "d" + str(row["duration"] - 1) + " "
                if activity:
                    story = add_action(story, row["n_activity"])
            else:
                trip_dur += row["duration"]

                if index == len(p_movement) - 1:
                    story += "t" + str(trip_dur + 1) + " "

        return story.rstrip()

    def add_action(story, n_action):
        if n_action > 0:
            story += "a1 "
        else:
            story += "a0 "
        return story

    def dwell_trip_action_construction(p_movement):
        return dwell_trip_construction(p_movement, True)

    func = {Construction.loc: location_construction,
            Construction.dwell_trip: dwell_trip_construction,
            Construction.dwell_trip_action: dwell_trip_action_construction}

    f_btw = []
    fb = []
    fa = []
    construction = func[c_name]

    for p in comps["comp"]:
        p_data = df[df["comp"] == p]

        if construction != location_construction:
            p_data = trip_detection(p_data)

        f_btw.append(construction(p_data))

        # TODO: use within-study analysis as optional
        middle = np.floor(len(p_data) / 2).astype(int)
        fb.append(construction(p_data.iloc[range(0, middle)]))
        fa.append(construction(p_data.iloc[range(middle, len(p_data))]))

    f_wth = (fb, fa)
    return f_btw, f_wth


def calc_sims(f, min_ngram, max_ngram, fb=None, fa=None):
    def cos_sim(features):
        # Method: cosine similarity (similar with repetition) + N-gram (time/sequence) + TF-IDF (normalization)
        vect = TfidfVectorizer(ngram_range=(min_ngram, max_ngram), norm='l2')
        tfidf = vect.fit_transform(features)

        return (tfidf * tfidf.T).A

    # Compare all movements between players
    sims = cos_sim(f)

    sims_within = []
    # Compare before and after movements within players
    if (fb and fa) is not None:
            for i in range(len(f)):
                if len(fb[i]) > 0 and len(fa[i]) > 0:
                    sims_within.append(cos_sim([fb[i], fa[i]])[0][1])
                else:
                    sims_within.append(-1)

    return sims, sims_within


def filter_sims(sims, participants, threshold):
    similars = []

    for i in range(0, len(sims)):
        for j in range(0, len(sims[i])):
            if j >= i:
                break

            score = sims[i][j]
            if score > threshold:
                similars.append((participants.index[i], participants.index[j], (1 - score)))

    return similars


def to_graph(sims, participants, threshold):
    similars = filter_sims(sims, participants, threshold)
    similars = sorted(similars, key=lambda x: x[2])

    G = nx.Graph()
    G.add_weighted_edges_from(similars)
    return G


def performance(f, c_name):
    process = psutil.Process(os.getpid())
    start_time = time.time()

    # lenghts = [len(x.split()) for x in f]

    sims = calc_sims(f, 1, c_name.value)

    print("--- STABLE --- Runtime (min): " + str((time.time() - start_time) / 60))
    print("--- STABLE --- Memory usage (GB): " + str(round(((process.memory_info().rss) / (10 ** 9)), 2)))

    return sims