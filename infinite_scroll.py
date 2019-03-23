import re
import time
import uuid
from lxml import html

from peewee import *
from selenium import webdriver

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from models import *


def main():
    chrome_path = r"C:/Users/seraz/Downloads/chromedriver_win32/chromedriver.exe"
    driver = webdriver.Chrome(chrome_path)

    Track.create_table()
    students = Student.select()
    # students = [Student.get(Student.name == 'Наташа')]
    for student in students:
        driver.get("https://vrit.me/audios{}".format(student.vk_id))

        lenOfPage = driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        match = False
        while match == False:
            lastCount = lenOfPage
            time.sleep(1)
            lenOfPage = driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
            if lastCount == lenOfPage:
                match = True

        text = driver.page_source
        tree = html.fromstring(text)
        titles = [title.text_content() for title in tree.xpath('//div[@class="info"]/div[@class="title"]')]
        authors = [artist.text_content() for artist in tree.xpath('//div[@class="info"]/div[@class="artist"]')]

        for t in zip(titles, authors):
            track = Track(id=uuid.uuid4(), title=t[0], student_id=student.id, author=t[1])
            track.save(force_insert=True)
        print(len(titles))


def create_db():
    Student.create_table()

    with open('ids.txt', 'r') as file:
        for line in file:
            x = line.split(' ')
            student = Student(id=uuid.uuid4(), name=' '.join(x[:-1]), vk_id=int(x[-1]))
            student.save(force_insert=True)


if __name__ == '__main__':
    # create_db()
    main()
