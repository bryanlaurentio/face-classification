# General utilities for use in image-handling operations
# Written by Glenn Jocher (glenn.jocher@ultralytics.com) for https://github.com/ultralytics

import os
from pathlib import Path
import requests
from PIL import Image

def download_uri(uri, dir='./'):
    f = dir + os.path.basename(uri) 
    with open(f, 'wb') as file:
        file.write(requests.get(uri, timeout=10).content)

    src = f  
    for c in ['%20', '%', '*', '~', '(', ')']:
        f = f.replace(c, '_')
    f = f[:f.index('?')] if '?' in f else f 
    if src != f:
        os.rename(src, f)  

    if Path(f).suffix == '':
        src = f  
        f += f'.{Image.open(f).format.lower()}'
        os.rename(src, f)  