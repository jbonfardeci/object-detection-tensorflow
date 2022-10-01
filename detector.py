import warnings
warnings.filterwarnings('ignore')
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.framework.ops import EagerTensor
from typing import List, Dict, Tuple, Any

class ObjectDetector:
    """
    ObjectDetector: A class for running object detection pipelines with pre-trained Tensorflow models.
    """
    class_list: List[str] = []
    color_list: List[Any] = []
    model_name: str = ''
    cache_dir: str = './pretrained_models'
    model: Any = None
    cache_subdir: str = 'checkpoints'

    def __init__(self, class_list: List[str], model_url: str, random_seed: int=42):
        """
        Initiate the object detection pipeline.
        
        @param class_list: List[str] - list of class names
        @param model_url: str - the URL of the pretrained model to download and use.
        @param random_seed: int = 42
        @returns ObjectDetector
        """
        np.random.seed(random_seed)
        self.class_list = class_list
        self.__set_color_list()
        self.download_model(model_url).load_model()

    @staticmethod
    def run(class_list: List[str], model_url: str, is_video: bool=False, **kwargs: Dict):
        """
        Runs a full object detection pipeline.

        @param class_list: List[str] - List of class names for labeling images.
        @param model_url: str - URL of the pretrained Tensorflow model.
        @param is_video: bool=False - if True, processes video, otherwise processes image.
        @param **kwargs: Dict - unpacks a dictionary object to pass to the respective video or image processor.
            Usage:

            For Images:
            ```
            image_args = {
                'filepath': './media/images/cat1.jpg',
                'output_path': './media/image_out/cat1.jpg',
                'write_file': True,
                'show_file': True,
                'threshold': 0.5, # lower number = more bounding boxes
                'max_output_size': 100, # higher number = more bounding boxes
                'line_thickness': 1,
                'draw_corners': False
            }
            ```

            For Video:
            ```
            video_args = {
                'filepath': './media/video/elephants.jpg',
                'output_path': './media/video_out/elephants.mp4',
                'write_file': True,
                'show_file': True,
                'threshold': 0.1, # lower number = more bounding boxes
                'max_output_size': 100, # higher number = more bounding boxes
                'line_thickness': 1,
                'draw_corners': False
            }
            ```

        @returns ObjectDetector
        """
        o = ObjectDetector(class_list, model_url).download_model(model_url).load_model()

        if is_video:
            o.predict_video(**kwargs)
        else:
            o.predict_image(**kwargs)
        return o

    def download_model(self, model_url: str):
        """
        Downloads and caches pretrained Tensorflow models.

        @param model_url: str
        @returns ObjectDetector
        """
        file_name: str = os.path.basename(model_url)
        self.model_name = file_name[:file_name.index('.')]
        os.makedirs(self.cache_dir, exist_ok=True)
        model_path = os.path.join(self.cache_dir, self.cache_subdir, file_name)
        if os.path.exists(model_path):
            print(f'The model {file_name} already exists, Skipping download.')
            return self
        get_file(
            fname=file_name,
            origin=model_url,
            cache_dir=self.cache_dir,
            cache_subdir=self.cache_subdir,
            extract=True
        )
        return self

    def load_model(self):
        print(f'Loading model {self.model_name}...')
        model_path = os.path.join(self.cache_dir, self.cache_subdir, self.model_name, 'saved_model')
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(model_path)
        print(f'Model {self.model_name} was loaded successfully.')
        return self

    def predict_image(
        self
        , filepath: str
        , output_path: str
        , write_file: bool=True
        , show_file: bool=False
        , threshold: float=0.5
        , max_output_size: int=50
        , line_thickness: int=4
        , draw_corners: bool=False
    ) -> np.ndarray:
        """
        Draws bounding boxes around detected objects in an image.

        @param filepath: str
        @param output_path: str
        @param write_file: bool=True
        @param show_file: bool=False
        @param threshold: float=0.5
        @param max_output_size: int=50
        @param line_thickness: int=4
        @param draw_corners: bool=False
        @returns np.ndarray - Image array
        """
        if not os.path.exists(filepath):
            raise Exception(f'{filepath} does not exist!')
            
        print(f'Opening {filepath}...')
        image: np.ndarray = cv2.imread(filepath)
        bbox_image: np.ndarray = self.__create_bounding_box(image, threshold, max_output_size, line_thickness, draw_corners)

        if show_file:
            cv2.imshow('Result', bbox_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if write_file:
            cv2.imwrite(filename=output_path, img=bbox_image)
            print(f'Wrote image to {output_path}.')

        return bbox_image

    def predict_video(
        self
        , filepath: str
        , output_path: str=None
        , write_file: bool=True
        , show_file: bool=False
        , threshold: float=0.5
        , max_output_size: int=50
        , line_thickness: int=4
        , draw_corners: bool=False
    ) -> None:
        """
        Draws bounding boxes around detected objects in a video.

        @param filepath: str
        @param output_path: str
        @param write_file: bool=True
        @param show_file: bool=False
        @param threshold: float=0.5
        @param max_output_size: int=50
        @param line_thickness: int=4
        @param draw_corners: bool=False
        @returns None
        """
        
        capture: cv2.VideoCapture = cv2.VideoCapture(filepath)

        if not capture.isOpened():
            raise Exception(f'Error opening video file at {filepath}')

        fps = float(30)
        height = int(capture.get(3))
        width = int(capture.get(4))

        if write_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output = cv2.VideoWriter(output_path, fourcc, fps, (height, width))

        start_time = 0
        ct = 0
        while(True):
            (ret, frame) = capture.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1/(current_time - start_time)
            start_time = current_time
            bbox_frame = self.__create_bounding_box(frame, threshold, max_output_size, line_thickness, draw_corners)
            cv2.putText(bbox_frame, 'FPS: ' + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            
            if show_file:
                cv2.imshow('Result', frame)
                # Press 'q' to escape.
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            if write_file:
                output.write(bbox_frame)

            ct += 1
            print(f'Reading frame {ct}.')
        # while

        if write_file:
            output.release()

        capture.release()
        return self

    def __set_color_list(self) -> None:
        self.color_list = np.random.uniform(low=0, high=255, size=(len(self.class_list), 3))

    def __create_bounding_box(
        self
        , image: np.ndarray
        , threshold: float=0.5
        , max_output_size: int=50
        , line_thickness: int=4
        , draw_corners: bool=False
    ) -> np.ndarray:

        input: np.ndarray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        input = tf.convert_to_tensor(input, dtype=tf.uint8)
        input = input[tf.newaxis,...]
        detections: Dict = self.model(input)
        boxes: np.ndarray = detections['detection_boxes'][0].numpy()
        class_indexes: List[int] = detections['detection_classes'][0].numpy().astype(np.int32)
        class_scores: np.ndarray = detections['detection_scores'][0].numpy()

        bbox_tensor: EagerTensor = tf.image.non_max_suppression(
            boxes=boxes,
            scores=class_scores,
            max_output_size=max_output_size,
            iou_threshold=threshold,
            score_threshold=threshold
        )

        if len(boxes) == 0:
            return image

        for i in bbox_tensor:
            box = tuple(boxes[i].tolist())
            confidence: float = round((100*class_scores[i]))
            class_ix: int = class_indexes[i]
            label_txt: str = self.class_list[class_ix]
            class_color = self.color_list[class_ix]
            display_txt = f'{label_txt}: {confidence}%'.upper()
            self.__draw_bounding_box(image, box, class_color, display_txt, line_thickness, draw_corners)
        # for
        return image

    def __draw_bounding_box(
        self
        , image: np.ndarray
        , box: Tuple
        , color: Any
        , text: str
        , line_thickness: int=4
        , draw_corners: bool=False
    ) -> None:

        height, width, channels = image.shape
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        ymin = int(ymin * height)
        ymax = int(ymax * height)

        cv2.rectangle(
            img=image,
            pt1=(xmin, ymin),
            pt2=(xmax, ymax),
            color=color,
            thickness=2
        )

        label_pos = (xmin, ymin - 12)

        cv2.putText(
            img=image,
            text=text,
            org=label_pos,
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=color,
            thickness=line_thickness
        )

        line_width = min(int((xmax-xmin)*0.2), int((ymax-ymin)*0.2))

        # draw corners on bounding box
        if not draw_corners:
            return
        thickness = line_thickness * 3
        # upper left horz
        cv2.line(image, (xmin, ymin), (xmin + line_width, ymin), color, thickness=thickness)
        # upper left vert
        cv2.line(image, (xmin, ymin), (xmin, ymin + line_width), color, thickness=thickness)

        # upper right horz
        cv2.line(image, (xmax, ymin), (xmax - line_width, ymin), color, thickness=thickness)
        # upper right vert
        cv2.line(image, (xmax, ymin), (xmax, ymin + line_width), color, thickness=thickness)

        # lower left horz
        cv2.line(image, (xmin, ymax), (xmin + line_width, ymax), color, thickness=thickness)
        # lower left vert
        cv2.line(image, (xmin, ymax), (xmin, ymax - line_width), color, thickness=thickness)

        # lower right horz
        cv2.line(image, (xmax, ymax), (xmax - line_width, ymax), color, thickness=thickness)
        # lower right vert
        cv2.line(image, (xmax, ymax), (xmax, ymax - line_width), color, thickness=thickness)
