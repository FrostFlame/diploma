import csv
import pickle
import numpy as np

from emotional_text_analysis.text_analises import predict
from models import *
from stats import my_cluster, cl


def main():
    model = pickle.load(open('models/Xgb, CharLevel Vectors', 'rb'))
    songs = {}
    with open('Lyrics1.csv', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        for row in reader:
            songs[(row[2], row[0])] = row[1]
    with open('Lyrics2.csv', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        for row in reader:
            songs[(row[2], row[0])] = row[1]
    students = Student.select()
    result = {}
    classes = []
    for student in students:
        student_lyrics = []
        tracks = Track.select().where(Track.student_id == student.id)
        for track in tracks:
            if (track.title, track.author) in songs.keys():
                student_lyrics.append(songs[(track.title, track.author)])
        if student_lyrics:
            l, classes = predict(model, student_lyrics)
            result[student.name] = l
    x = {}
    for e in result.keys():
        d = {}
        for cla in classes:
            d[cla] = 0
        for i in result[e]:
            d[i] += 1
        s = sum(d.values())
        for i in d.keys():
            d[i] = d[i] / s
        x[e] = list(d.values())

    my_cluster(np.array(list(x.values())), np.array(list(x.keys())))
    cl(np.array([[e[7], e[8]] for e in list(x.values())]), np.array(list(x.keys())), threshold=0.03)
    for e in x.keys():
        print(e + ': ' + '\t\t\t'.join(str(i) for i in (x[e])))


if __name__ == '__main__':
    main()

#                   anger boredom empty     enthusiasm      fun     happiness       hate    love      neutral      relief      sadness      surprise    worry
# Амир:             0.0		0.0	   0.0			0.0			0.034	    0.0			0.0		0.27	   0.66			0.0			0.0			0.0			0.025
# Тимур Тимерханов: 0.0		0.0	   0.0			0.0			0.01		0.0			0.0		0.23	   0.72			0.0			0.005		0.0			0.02
# Тая:              0.0		0.0	   0.003		0.0			0.01		0.0			0.0		0.27	   0.71			0.003		0.0			0.0			0.003
# МА:               0.0		0.0	   0.0			0.0			0.0			0.0			0.0		0.28	   0.71			0.0			0.0			0.0			0.0
# Тимур Марданов:   0.0		0.0	   0.0			0.0			0.0			0.0			0.0		0.45	   0.53			0.0			0.0			0.0			0.009
# Наташа:           0.0		0.0	   0.0			0.0			0.0			0.0			0.0		0.28	   0.71			0.0			0.0			0.0			0.0
# Света:            0.0		0.0	   0.0			0.0			0.0			0.0			0.0		0.0		   1.0			0.0			0.0			0.0			0.0
# Сагит:            0.0		0.0	   0.019		0.0			0.02		0.0			0.0		0.21	   0.74			0.0			0.0			0.0			0.0
# Марат:            0.0		0.0	   0.0			0.0			0.0			0.0			0.0		0.23	   0.75			0.019		0.0			0.0			0.0
# Дамир:            0.0		0.0	   0.0			0.0			0.0			0.0			0.0		0.14	   0.85			0.0			0.0			0.0			0.0
# Екатерина:        0.0		0.0	   0.004		0.0			0.0			0.0			0.0		0.23	   0.75			0.0			0.0			0.004		0.004
