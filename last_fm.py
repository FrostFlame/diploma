import time
from urllib import error
from urllib.parse import quote
from urllib.request import urlopen
from bs4 import BeautifulSoup

from infinite_scroll import Track, Tags

url = 'http://ws.audioscrobbler.com/2.0/'
api_key = 'e136ab596c173fcb97abd64833bb276a'


def get_tags():
    method = 'track.gettoptags'
    t = time.time()
    Tags.create_table()
    tracks = Track.select()
    for track in tracks:
        t = Tags.select().where(Tags.track_id == track.id)
        if not t:
            try:
                x = '{}?method={}&api_key={}&artist={}&track={}'.format(url, method, api_key, quote(track.author),
                                                                        quote(track.title))
                response = urlopen(x).read()
                soup = BeautifulSoup(response, 'html.parser')
                tags = soup.find_all('tag')
                if tags:
                    for elem in tags:
                        if int(elem.find('count').get_text()) > 60:
                            tag = Tags(track_id=track.id, name=elem.find('name').get_text())
                            tag.save()
            except error.HTTPError:
                pass
    print(time.time() - t)


def get_similar_tag(tag):
    method = 'tag.getsimilar'
    try:
        x = '{}?method={}&tag={}&api_key={}'.format(url, method, tag.lower(), api_key)
        response = urlopen(x).read()
        soup = BeautifulSoup(response, 'html.parser')
        tags = soup.find_all('tag')
        if tags:
            return [elem.name for elem in tags]
        else:
            return []
    except error.HTTPError:
        pass


if __name__ == '__main__':
    get_tags()
