# Written by Glenn Jocher (glenn.jocher@ultralytics.com) for https://github.com/ultralytics
import argparse
import os
import time

from flickrapi import FlickrAPI
from utilsFlickr.general import download_uri

key = '3186e3fba9a93bdffec3acc7724b0a4e'  # Flickr API key https://www.flickr.com/services/apps/create/apply
secret = '82b3c69df9555877'

def get_urls(search='honeybees on flowers', n=100, download=False):
    t = time.time()
    flickr = FlickrAPI(key, secret)
    license = ()  
    photos = flickr.walk(text=search, 
                         extras='url_o',
                         per_page=500,  
                         license=license,
                         sort='relevance')

    if download:
        dir = os.getcwd() + os.sep + 'images' + os.sep + search.replace(' ', '_') + os.sep  # save directory
        if not os.path.exists(dir):
            os.mkdir(dir)

    urls = []
    for i, photo in enumerate(photos):
        if i < n:
            try:
                url = photo.get('url_o')  
                if url is None:
                    url = f"https://farm{photo.get('farm')}.staticflickr.com/{photo.get('server')}/{photo.get('id')}_{photo.get('secret')}_b.jpg"
                if download:
                    download_uri(url, dir)
                urls.append(url)
                print('%g/%g %s' % (i, n, url))
            except:
                print('%g/%g error...' % (i, n))
    print('Done. (%.1fs)' % (time.time() - t) + ('\nAll images saved to %s' % dir if download else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', type=str, default='honeybees on flowers', help='flickr search term')
    parser.add_argument('--n', type=int, default=1, help='number of images')
    parser.add_argument('--download', action='store_true', help='download images')
    opt = parser.parse_args()

    help_url = 'https://www.flickr.com/services/apps/create/apply'
    assert key and secret, f'Flickr API key required in flickr_scraper.py L11-12. To apply visit {help_url}'

    get_urls(search=opt.search, 
             n=opt.n, 
             download=opt.download)  