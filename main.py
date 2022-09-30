import tensorflow as tf
import os
from typing import List


def get_files(dir) -> List[str]:
    paths = []
    for dir, _, files in os.walk(dir):
        paths.extend([os.path.join(dir, fn) for fn in files])
    return paths


def get_class_names(filepath) -> List[str]:
    with open(filepath, 'r') as f:
        return f.read().splitlines()


if __name__ == '__main__':
    print(tf.version.VERSION)

    os.environ['CUDA_VISIBLE_DEVICES'] ="0"

    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow **IS** using the GPU")
    else:
        print("TensorFlow **IS NOT** using the GPU")

    # Pre-trained Tensorflow models
    # Add more model links to this dictionary from this URL:
    #    All models list: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    models = {
        'ssd_mobilenet': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        'efficientdet_d4': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz',
        'faster_rcnn': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz',
        'mask_rcnn_inception_resnet': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz',
        'efficientdet_d7': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz',
        'centernet_mobilenet': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz'
    }

    selected_model = models['mask_rcnn_inception_resnet']

    media_dir = './media'
    image_dir = f'{media_dir}/images'
    video_dir = f'{media_dir}/video'
    image_out_dir = f'{media_dir}/image_out'
    video_out_dir = f'{media_dir}/video_out'
    class_names = get_class_names('./coco.names')

    video_paths = get_files(video_dir)
    image_paths = get_files(image_dir)

    img_args = {
        'filepath': f'{image_dir}/mountain-bikes.jpg',
        'output_path': f'{image_out_dir}/mountain-bikes.jpg',
        'write_file': True,     # Save rendered file to folder.
        'show_file': True,      # Show realtime rendering.
        'threshold': 0.5,       # lower number = more bounding boxes
        'max_output_size': 100, # higher number = more bounding boxes
        'line_thickness': 1,    # Thickness of bounding box line.
        'draw_corners': False   # Emphasize corners on bounding box.
    }

    video_args = {
        'filepath': f'{video_dir}/gallion.mp4',
        'output_path': f'{video_out_dir}/gallion.mp4',
        'write_file': True,     # Save rendered file to folder.
        'show_file': True,      # Show realtime rendering.
        'threshold': 0.1,       # lower number = more bounding boxes
        'max_output_size': 100, # higher number = more bounding boxes
        'line_thickness': 1,    # Thickness of bounding box line.
        'draw_corners': False   # Emphasize corners on bounding box.
    }

    from detector import ObjectDetector

    detector = ObjectDetector(class_names, selected_model)

    # Process video
    detector.predict_video(**video_args)

    # Process Image
    detector.predict_image(**img_args)
    
    # Process image batch
    img_args['show_file'] = False
    for image_path in image_paths:
        img_args['filepath'] = image_path
        filename = os.path.basename(image_path)
        img_args['output_path'] = f'{image_out_dir}/{filename}'
        detector.predict_image(**img_args)
    # for