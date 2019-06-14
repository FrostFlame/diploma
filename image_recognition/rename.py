import operator
import os
import pickle
import time
from distutils.version import LooseVersion

import face_recognition
import matplotlib
import numpy as np
from scipy.interpolate import spline, make_interp_spline
from scipy.stats import norm
from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from IPython.display import HTML

from image_recognition.test import instanames
from stats import my_cluster, cl


def main():
    known_images = '/home/damir/coding/insta/known'
    known_faces = {}
    for file in os.listdir(known_images):
        try:
            picture = face_recognition.load_image_file(known_images + '/' + file)
            face_encoding = face_recognition.face_encodings(picture)[0]
            known_faces[file.split('.')[0]] = face_encoding
        except IndexError:
            pass

    unknown_faces = {}
    count_all = {} #ключи - имена папок
    for root, subdirs, _ in os.walk('/home/damir/coding/insta/scrape'):
        for dir in subdirs:
            i = 0
            y = 0
            for file in os.listdir(root + '/' + dir):
                try:
                    picture = face_recognition.load_image_file(root + '/' + dir + '/' + file)
                    face_encoding = face_recognition.face_encodings(picture)[0]
                    # i += 1
                    unknown_faces[file] = face_encoding
                except IndexError:
                    # print('{}: не удалось найти лица'.format(file))
                    pass
                y += 1
            count_all[dir] = y
            print(dir + ' ' + str(y))
    with open('unknown_images.dump', 'wb') as f:
        pickle.dump(unknown_faces, f)
    with open('unknown_images.dump', 'rb') as f:
        unknown_faces = pickle.load(f)
    # count_all = dict(zip(list(known_faces.keys()), count_all))
    count_portrets = dict(zip(list(known_faces.keys()), [0] * len(known_faces.keys())))
    for image_name, un_face in unknown_faces.items():
        results = face_recognition.compare_faces(list(known_faces.values()), un_face)
        if True in results:
            index = results.index(True)
            name = list(known_faces.keys())[index]
            tuple = [e for e in instanames if name in e][0]
            if tuple[1] in image_name:
                count_portrets[name] += 1
            # print('{}: {}'.format(image_name, list(known_faces.keys())[index]))
        # else:
            # print('{}: {}'.format(image_name, 'Не опознано'))
    values = []
    for name in known_faces.keys():
        values.append(count_portrets[name] / count_all[[e for e in instanames if name in e][0][1]])
        print('{}: {}'.format(name, count_all[[e for e in instanames if name in e][0][1]]))
        print('{}: {}    {}   {}'.format(name, count_all[[e for e in instanames if name in e][0][1]], count_portrets[name], count_portrets[name] / count_all[[e for e in instanames if name in e][0][1]]))

    X = values
    labels = []
    for name in count_all.keys():
        labels.append([e for e in instanames if name in e][0][0])
        # labels.append(name)
    X = dict(zip(labels, X))
    sorted_x = sorted(X.items(), key=operator.itemgetter(1))
    my_xticks = np.array([i[0] for i in sorted_x])

    T = np.array([e for e in range(len(my_xticks))])
    power = [i[1] for i in sorted_x]
    xnew = np.linspace(T.min(), T.max(), 300)
    spl = make_interp_spline(T, power, k=1)
    power_smooth = spl(xnew)

    plt.xticks(T, my_xticks)
    plt.plot(xnew, power_smooth)
    plt.show()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print((end - start) / 60)
# kate: 25    4   0.16
# sveta: 15    14   0.9333333333333333
# michael: 14    6   0.42857142857142855
# marat: 17    5   0.29411764705882354
# taya: 14    6   0.42857142857142855
# damirm: 25    4   0.16
# natasha: 40    9   0.225
# amir: 14    1   0.07142857142857142
# sagit: 43    2   0.046511627906976744