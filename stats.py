from sklearn.cluster import DBSCAN

from last_fm import get_similar_tag
from models import *
import sklearn
import numpy as np
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt


def cl(X, names, threshold):
    db = DBSCAN(eps=threshold, min_samples=1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # #############################################################################
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    for point, name in zip(X, names):
        plt.text(point[0]+0.03, point[1]+0.05, name, fontsize=9)

    plt.title('Estimated number of clusters: %d' % n_clusters_)

    for name, label in zip(names, labels):
        print(name + ' ' + str(label))
    plt.show()


def get_dict(check_similar=False):
    students = Student.select(Student.name)
    d = {}
    for s in students:
        tags = [e.name for e in Tags.select().join(Track).join(Student).where(Student.name == s.name)]
        if not tags:
            continue
        l = []
        for g in genres:
            i = 0
            similar = get_similar_tag(g) if check_similar else []
            for t in tags:
                if g.lower() == t.lower() or t.lower() in similar:
                    i += 1
            l.append(i)
        d[s.name] = l
    return d


def pure():
    d = get_dict()
    X = np.array(list(d.values()))
    labellist = list(d.keys())
    plt.figure(figsize=(11, 7))
    dend = shc.dendrogram(shc.linkage(X, method='ward'), labels=np.array(labellist))
    plt.show()
    for key in d.keys():
        print(key + ': ' + str(d[key]))


def my_cluster(X, labels):
    plt.figure(figsize=(15, 7))
    dend = shc.dendrogram(shc.linkage(X, method='ward'), labels=labels)
    plt.show()


def preferences():
    d = get_dict()
    print(d)
    psychos = Psycho.select()
    prefs = Genres.select()
    prefs_vectors = []
    psycho_vectors = []
    for student in d.keys():
        st_dict = dict(zip(genres, d[student]))
        s_ps = {}
        for ps in ['mellow', 'unpretentious', 'sophisticated', 'intense', 'contemporary']:
            i = 0
            for p in prefs:
                i += getattr(p, ps) * st_dict[p.name]
            i = i / len(prefs)
            s_ps[ps] = i
        print(student + ' ' + str(s_ps))
        prefs_vectors.append(list(s_ps.values()))
        psychos_dict = {}
        for psycho in psychos:
            i = 0
            for key in s_ps.keys():
                i += s_ps[key] * getattr(psycho, key)
            i = i / len(s_ps.keys())
            psychos_dict[psycho.name] = i
        psycho_vectors.append(list(psychos_dict.values()))
        print(student + ' ' + str(psychos_dict))
        print('**********************')
    my_cluster(np.array(prefs_vectors), np.array(list(d.keys())))
    my_cluster(np.array(psycho_vectors), np.array(list(d.keys())))
    my_cluster(np.array([[e[2], e[3]] for e in prefs_vectors]), np.array(list(d.keys())))
    cl(np.array([[e[2], e[3]] for e in prefs_vectors]), np.array(list(d.keys())), 0.17)


if __name__ == '__main__':
    # pure()
    preferences()
