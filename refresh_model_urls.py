from bs4 import BeautifulSoup
import utilities as u
import re

"""
Run this file to download all Tensorflow model names and URLs 
and store them in a JSON file in the './resources' directory.
"""


if __name__ == '__main__':
    url = 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md'
    models_html = u.get_html(url, './resources/models.html')
    soup = BeautifulSoup(models_html, 'html.parser')
    elems = soup.select('div#readme > article > table > tbody > tr > td > a')
    models = {}
    for elem in elems:
        model_name = re.sub(r'\s+', '_', elem.text.lower())
        models[model_name] = elem['href']

    u.write_json(models, './resources/models.json', 4)