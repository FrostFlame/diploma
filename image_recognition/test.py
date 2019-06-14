import os
import subprocess

instanames = [('natasha', 'weil3_fuchs'),
              ('amir', 'kadami_'),
              ('taya', 'soft.fox'),
              ('kate', 'alpha._.centauri'),
              ('nonka', 'uusermeme'),
              ('damirm', 'neuro_duck'),
              ('michael', 'abramichael'),
              ('nekit', 'zuzyk_'),
              ('sveta', 'basargina__'),
              ('sagit', 'snakeaya'),
              ('marat', 'hey_solncev'),
              ('leysan', 'leysan_san')
              ]


def main():
    for name, link in instanames.items():
        p = subprocess.Popen(['instagram-scraper', link, '-u', 'frost._.flame', '-p', 'instagramvzlomchik'],
                             cwd='/home/damir/coding/insta/scrape')
        p.wait()
        for r, d, f in os.walk('/home/damir/coding/insta/scrape/' + link):
            for file in f:
                if not file.endswith('.jpg'):
                    os.remove('/home/damir/coding/insta/scrape/' + link + '/' + file)


def rename():
    for link in instanames:
        for r, d, f in os.walk('/home/damir/coding/insta/scrape/' + link):
            i = 1
            for file in f:
                os.rename('/home/damir/coding/insta/scrape/' + link + '/' + file,
                          '/home/damir/coding/insta/scrape/' + link + '/' + link + '_' + str(i) + '.jpg')
                i += 1


if __name__ == '__main__':
    # main()
    rename()
