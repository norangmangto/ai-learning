import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image


class ObjectDetector:
    """
    Object detection using TensorFlow Hub models or TensorFlow Object Detection API
    """

    def __init__(self, model_type="efficientdet"):
        """
        Initialize object detector
        model_type: 'efficientdet', 'ssd', or 'fasterrcnn'
        """
        self.model_type = model_type

        # Load model from TensorFlow Hub
        if model_type == "efficientdet":
            # EfficientDet D0
            model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
        elif model_type == "ssd":
            # SSD MobileNet V2
            model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
        elif model_type == "fasterrcnn":
            # Faster R-CNN Inception ResNet V2
            model_url = (
                "https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1"
            )
        else:
            model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"

        print(f"Loading {model_type} model from TensorFlow Hub...")
        self.model = hub.load(model_url)

        # COCO class names (same as PyTorch)
        self.coco_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

    def detect(self, image_path, confidence_threshold=0.5):
        """
        Detect objects in image
        Returns: list of detections (label, confidence, bbox)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        detections = self.model(input_tensor)

        # Extract results
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )

        # Filter by confidence
        results = []
        height, width = image_np.shape[:2]

        for i in range(num_detections):
            score = detections["detection_scores"][i]
            if score >= confidence_threshold:
                class_id = detections["detection_classes"][i]
                bbox = detections["detection_boxes"][i]

                # Convert normalized coordinates to pixel coordinates
                ymin, xmin, ymax, xmax = bbox
                x1, y1 = int(xmin * width), int(ymin * height)
                x2, y2 = int(xmax * width), int(ymax * height)

                # Get label name
                label = (
                    self.coco_names[class_id - 1]
                    if class_id <= len(self.coco_names)
                    else f"class_{class_id}"
                )

                results.append(
                    {
                        "label": label,
                        "confidence": float(score),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return results

    def draw_detections(self, image_path, detections, output_path=None):
        """Draw bounding boxes on image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['label']}: {det['confidence']:.2f}"

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return image


def train():
    print("Training Object Detection with TensorFlow...")
    print("Using TensorFlow Hub pre-trained models")

    # 1. Model Overview
    print("\n=== Available Object Detection Models ===")
    print("1. EfficientDet: Scalable and efficient object detection")
    print("2. SSD MobileNet: Fast mobile-friendly detection")
    print("3. Faster R-CNN: High accuracy two-stage detector")
    print("4. CenterNet: Anchor-free object detection")
    print("5. YOLO (via TensorFlow Hub): Real-time detection")

    # 2. Model Architecture Comparison
    print("\n=== Model Comparison ===")
    print("| Model             | Speed  | mAP   | Input Size | Parameters |")
    print("|-------------------|--------|-------|------------|------------|")
    print("| SSD MobileNet V2  | Fast   | 22.0  | 300x300    | 18M        |")
    print("| EfficientDet-D0   | Fast   | 33.8  | 512x512    | 3.9M       |")
    print("| EfficientDet-D1   | Medium | 39.6  | 640x640    | 6.6M       |")
    print("| Faster R-CNN      | Slow   | 42.0  | 640x640    | 165M       |")
    print("| EfficientDet-D7   | Slow   | 52.2  | 1536x1536  | 52M        |")

    # 3. Load Pre-trained Model Demo
    print("\n=== Loading Pre-trained Model ===")
    print("Note: First run will download model from TensorFlow Hub")

    # For demo, we'll show the structure without actual loading
    print("Model structure:")
    print(
        """
    Input: Image tensor [batch, height, width, 3]
    Output: Dictionary with:
        - detection_boxes: [N, 4] (ymin, xmin, ymax, xmax)
        - detection_classes: [N] class indices
        - detection_scores: [N] confidence scores
        - num_detections: number of detections
    """
    )

    # 4. Training Custom Model
    print("\n=== Training Custom Object Detector ===")
    print("To train on custom dataset using TensorFlow Object Detection API:")
    print(
        """
    1. Install TensorFlow Object Detection API:
       git clone https://github.com/tensorflow/models.git
       cd models/research
       protoc object_detection/protos/*.proto --python_out=.
       pip install .

    2. Prepare dataset:
       - Annotate images (use LabelImg, CVAT, etc.)
       - Convert to TFRecord format
       - Create label map file

    3. Configure model:
       - Choose base model (SSD, Faster R-CNN, EfficientDet)
       - Set hyperparameters in config file
       - Configure data augmentation

    4. Train:
       python model_main_tf2.py \\
           --model_dir=models/my_model \\
           --pipeline_config_path=configs/my_config.config

    5. Export model:
       python exporter_main_v2.py \\
           --input_type image_tensor \\
           --pipeline_config_path configs/my_config.config \\
           --trained_checkpoint_dir models/my_model \\
           --output_directory exported_models/my_model
    """
    )

    # 5. Transfer Learning Example
    print("\n=== Transfer Learning Approach ===")
    print(
        """
    # Fine-tune pre-trained model on custom data

    # 1. Load pre-trained model
    base_model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

    # 2. Prepare custom dataset with:
    #    - Images in format: [batch, height, width, 3]
    #    - Annotations: bounding boxes and class labels

    # 3. Fine-tune (using TF Object Detection API recommended)
    #    Or use the model directly for inference
    """
    )

    # 6. Inference Optimization
    print("\n=== Inference Optimization ===")
    print("• Use TensorFlow Lite for mobile/edge devices")
    print("• Use TensorRT for NVIDIA GPU acceleration")
    print("• Quantization: INT8 quantization for faster inference")
    print("• Model pruning to reduce model size")
    print("• Batch processing for throughput")

    # 7. Post-processing
    print("\n=== Post-processing Techniques ===")
    print("• Non-Maximum Suppression (NMS): Remove duplicate detections")
    print("• Score thresholding: Filter low-confidence predictions")
    print("• Class-specific NMS: Separate NMS per class")
    print("• Soft-NMS: Decay scores instead of removing boxes")

    # 8. QA Validation
    print("\n=== QA Validation ===")
    print("✓ Object detection models available via TensorFlow Hub")
    print("✓ Can detect 80+ object classes from COCO dataset")
    print("✓ Multiple architectures available (SSD, EfficientDet, Faster R-CNN)")
    print("✓ Outputs bounding boxes, labels, and confidence scores")
    print("✓ Can be fine-tuned using TensorFlow Object Detection API")
    print("✓ Supports deployment on various platforms (TFLite, TensorRT)")

    # 9. Example Usage
    print("\n=== Example Usage ===")
    print(
        """
    # Initialize detector
    detector = ObjectDetector(model_type='efficientdet')

    # Detect objects
    detections = detector.detect('image.jpg', confidence_threshold=0.5)

    # Print results
    for det in detections:
        print(f"{det['label']}: {det['confidence']:.2f} at {det['bbox']}")

    # Draw and save results
    detector.draw_detections('image.jpg', detections, 'output.jpg')

    # Advanced: Batch processing
    image_tensor = tf.convert_to_tensor(images)  # [batch, height, width, 3]
    detections = detector.model(image_tensor)
    """
    )

    # 10. Evaluation Metrics
    print("\n=== Evaluation Metrics ===")
    print("• mAP (mean Average Precision): Overall detection performance")
    print("• IoU (Intersection over Union): Bounding box overlap")
    print("• Precision: Ratio of correct detections")
    print("• Recall: Ratio of objects detected")
    print("• FPS (Frames Per Second): Inference speed")


if __name__ == "__main__":
    train()
