
MEDIA_DIR = 'MEDIA_DIR'
IMAGE_DIR = 'IMAGE_DIR'
IMAGE_OUT_DIR = 'IMAGE_OUT_DIR'
VIDEO_DIR = 'VIDEO_DIR'
VIDEO_OUT_DIR = 'VIDEO_OUT_DIR'
COCO_NAMES = 'COCO_NAMES'
MODELS_JSON = 'MODELS_JSON'
MODELS_TXT = 'MODELS_TXT'

def get_path(path) -> str:
    media_dir = './media'
    p = {}
    p[MEDIA_DIR] = media_dir
    p[IMAGE_DIR] = f'{media_dir}/images'
    p[VIDEO_DIR] = f'{media_dir}/video'
    p[IMAGE_OUT_DIR] = f'{media_dir}/image_out'
    p[VIDEO_OUT_DIR] = f'{media_dir}/video_out'
    p[COCO_NAMES] = './coco.names'
    p[MODELS_JSON] = './resources/models.json'
    p[MODELS_TXT] = './resources/models.txt'
    
    if path in p:
        return p[path]
    raise Exception(f"Path {path} was not found.")