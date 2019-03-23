import uuid

from peewee import *

db = PostgresqlDatabase('Diploma', user='postgres', password='postgres', host='localhost', port=5432)
genres = ['Soft rock', 'R&B/soul', 'Quiet storm', 'Adult contemporary', 'Rap', 'Electronica', 'Latin', 'Acid jazz',
          'Euro pop', 'Classical', 'Marching band', 'Avant-garde classical', 'Polka', 'World beat', 'Traditional jazz',
          'Celtic', 'Classic rock', 'Punk', 'Heavy metal', 'Power pop', 'Alternative rock', 'New country',
          'Mainstream country', 'Country rock', 'Bluegrass', 'Rock-n-roll']
genres = [e.lower() for e in genres]


class Student(Model):
    id = UUIDField(primary_key=True)
    name = CharField(max_length=32)
    vk_id = IntegerField()

    class Meta:
        database = db
        db_table = 'students'


class Track(Model):
    id = UUIDField(primary_key=True)
    title = CharField()
    student_id = ForeignKeyField(Student, to_field='id', db_column='student_id')
    author = CharField()

    class Meta:
        database = db
        db_table = 'tracks'


class Tags(Model):
    track_id = ForeignKeyField(Track, to_field='id', db_column='track_id')
    name = CharField()

    class Meta:
        database = db
        db_table = 'tags'


class Psycho(Model):
    id = UUIDField(primary_key=True)
    name = CharField()
    mellow = FloatField()
    unpretentious = FloatField()
    sophisticated = FloatField()
    intense = FloatField()
    contemporary = FloatField()

    class Meta:
        database = db
        db_table = 'psycho'


class Genres(Model):
    id = UUIDField(primary_key=True)
    name = CharField()
    mellow = FloatField()
    unpretentious = FloatField()
    sophisticated = FloatField()
    intense = FloatField()
    contemporary = FloatField()

    class Meta:
        database = db
        db_table = 'genres'


if __name__ == '__main__':
    Psycho.create_table()
    for i in range(7):
        ps = Psycho(id=uuid.uuid4(), name='', mellow=0, unpretentious=0, sophisticated=0, intense=0, contemporary=0)
        ps.save(force_insert=True)
