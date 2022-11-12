import tensorflow as tf
import os
from typing import List, Dict
import utilities as u
from ObjectDetector import ObjectDetector
import paths


def env_config() -> None:
    """Check Tensorflow environment."""
    print(tf.version.VERSION)

    os.environ['CUDA_VISIBLE_DEVICES'] ="0"

    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow **IS** using the GPU")
    else:
        print("TensorFlow **IS NOT** using the GPU")
    
    
def get_detector(model_name: str=None) -> ObjectDetector:
    """
    Get a configured instance of ObjectDetector
    @param model_name: str (Default is 'faster_r_cnn_inception_resnet_v2_1024x1024')
    @returns ObjectDetector
    """
    if model_name is None:
        model_name = 'faster_r_cnn_inception_resnet_v2_1024x1024'
        
    # Load the class names for our detected objects.
    class_names = u.get_file_data(paths.get_path(paths.COCO_NAMES))

    # Load the pre-trained Tensorflow models from JSON.
    models: Dict = u.get_json(paths.get_path(paths.MODELS_JSON))

    # Load your chosen model for object detection.
    selected_model = models[model_name]

    # Setup the object detection pipeline.
    return ObjectDetector(class_names, selected_model, 777)
    
    
def detect_image(args: Dict, model_name: str=None) -> None:
    """
    Draw bounding boxes and label objects in an image.
    @param args: Dict
    @param model_name: str (Optional)
    @returns None
    """
    u.check_filepath(args['filepath'])
    detector = get_detector(model_name)
    detector.predict_image(**args)


def detect_video(args: Dict, model_name: str=None) -> None:
    """
    Draw bounding boxes and label objects in a video.
    @param args: Dict
    @param model_name: str (Optional)
    @returns None
    """
    u.check_filepath(args['filepath'])
    detector = get_detector(model_name)
    detector.predict_video(**args)


if __name__ == '__main__':
    """Entry Point"""
    env_config()

    image_dir = paths.get_path(paths.IMAGE_DIR)
    video_dir = paths.get_path(paths.VIDEO_DIR)
    image_out_dir = paths.get_path(paths.IMAGE_OUT_DIR)
    video_out_dir = paths.get_path(paths.VIDEO_OUT_DIR)

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

    # See `./resources/models.json` for all available models.
    model_name = 'faster_r_cnn_resnet50_v1_640x640'
    
    # Arguments for detecting objects in an image.
    img_filename = 'mountain-biker.png'
    img_args = base_args.copy()
    img_args['filepath'] = f'{image_dir}/{img_filename}'
    img_args['output_path'] = f'{image_out_dir}/{img_filename}'
    detect_image(img_args, model_name)
    
    # Arguments for detecting objects in a video.
    video_filename = 'dog-on-boat.mp4'
    video_args = base_args.copy()
    video_args['filepath'] = f'{video_dir}/{video_filename}'
    video_args['output_path'] = f'{video_out_dir}/{video_filename}'
    detect_video(video_args, model_name)