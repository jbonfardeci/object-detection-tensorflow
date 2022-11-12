import os
import json
import re
from urllib3 import PoolManager
from urllib3.response import HTTPResponse
import urllib3 as urllib
from typing import List, Dict, Any

def check_filepath(filepath) -> None:
    if not os.path.exists(filepath):
        raise Exception(f'{filepath} does not exist!')


def get_file(filepath: str) -> str:
    check_filepath(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(content: str, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(content)


def get_file_list(dir) -> List[str]:
    paths = []
    for dir, _, files in os.walk(dir):
        paths.extend([os.path.join(dir, fn) for fn in files])
    return paths


def get_file_data(filepath) -> List[str]:
    return get_file(filepath).splitlines()


def get_json(filepath: str) -> Dict:
    file = get_file(filepath)
    return json.loads(file)


def write_json(data: Any, filepath: str, indent: int=None) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def get_html(url: str, output_path: str, force: bool=False) -> str:
    """
    Gets the raw HTML from a web page URL.
    @param url (str)
    @returns html (str)
    """
    if os.path.exists(output_path) and not force:
        return get_file(output_path)
    try:
        http: PoolManager = urllib.PoolManager()
        res: HTTPResponse = http.request('GET', url)
        html = res.data.decode('utf-8')
        write_file(html, output_path)
        return html
    except Exception as e:
        raise e