import tensorflow as tf
import os
from typing import List, Dict
import utilities as u

def env_config():
    print(tf.version.VERSION)

    os.environ['CUDA_VISIBLE_DEVICES'] ="0"

    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow **IS** using the GPU")
    else:
        print("TensorFlow **IS NOT** using the GPU")


if __name__ == '__main__':
    env_config()

    media_dir = './media'
    image_dir = f'{media_dir}/images'
    video_dir = f'{media_dir}/video'
    image_out_dir = f'{media_dir}/image_out'
    video_out_dir = f'{media_dir}/video_out'

    # Load the class names for our detected objects.
    class_names = u.get_file_data('./coco.names')

    # Load the pre-trained Tensorflow models from JSON.
    models: Dict = u.get_json('./resources/models.json')

    # Load your chosen model for object detection.
    selected_model = models['mask_r-cnn_inception_resnet_v2_1024x1024']

    # Get list of videos from the media folder.
    video_paths = u.get_file_list(video_dir)

    # Get list of images from the media folder.
    image_paths = u.get_file_list(image_dir)

    base_args = {
        'filepath': None,       # Image/video to detect.
        'output_path': None,    # Where to output the results.
        'write_file': True,     # Save rendered file to folder.
        'show_file': True,      # Show realtime rendering.
        'threshold': 0.5,       # lower number = more bounding boxes
        'max_output_size': 100, # higher number = more bounding boxes
        'line_thickness': 1,    # Thickness of bounding box line.
        'draw_corners': False   # Emphasize corners on bounding box.
    }

    # Arguments for detecting objects in images.
    img_args = base_args.copy()
    img_args['filepath'] = f'{image_dir}/mountain-bikes.jpg'
    img_args['output_path'] = f'{image_out_dir}/mountain-bikes.jpg'

    # Arguments for detecting objects in videos.
    video_args = base_args.copy()
    video_args['filepath'] = f'{video_dir}/gallion.mp4'
    video_args['output_path'] = f'{video_out_dir}/gallion.mp4'

    # Setup the object detection pipeline.
    from detector import ObjectDetector

    detector = ObjectDetector(class_names, selected_model, 777)

    # Process an image.
    detector.predict_image(**img_args)

    # Process a video.
    detector.predict_video(**video_args)

    # Process a batch of images.
    img_args['show_file'] = False
    for image_path in image_paths:
        img_args['filepath'] = image_path
        filename = os.path.basename(image_path)
        img_args['output_path'] = f'{image_out_dir}/{filename}'
        detector.predict_image(**img_args)
    # for