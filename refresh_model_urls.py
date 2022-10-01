import utilities as u
import re
from typing import List, Dict

"""
Run this file to download all Tensorflow model names and URLs 
and store them in a JSON file in the './resources' directory.
"""

if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/g3doc/tf2_detection_zoo.md'
    filename = './resources/models.txt'
    models_text = u.get_html(url, filename)
    model_data: List[str] = u.get_file_data(filename)
    start_index = [i for i in range(len(model_data)) if re.match(r'-+', model_data[i])][0]+1
    models = {}

    for line in model_data[start_index:]:
        model = re.findall(r'^\[.+\]\(.+\)', line)[:1]
        if model:
            model_name, model_url = model[0].split(']')
            model_name = re.sub(r'[^a-z0-9]', '_', model_name[1:].lower()).strip()
            model_url = re.sub(r'\(|\)', '', model_url).strip()
            models[model_name] = model_url

    if len(models.keys()) > 0:
        u.write_json(models, './resources/models.json', 4)