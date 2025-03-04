import cv2
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class YoloDetector:
    """
    Object detector using YOLOv4-tiny
    """

    def __init__(
        self,
        config_path,
        weights_path,
        labels_path,
        conf_threshold=0.5,
        nms_threshold=0.4,
        draw_bboxes=True,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.draw_bboxes = draw_bboxes

        # Load class labels
        with open(labels_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255, size=(len(self.class_names), 3), dtype=np.uint8
        )

        # Load the network
        logger.info("Loading YOLO model...")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using CUDA backend")
        except Exception as e:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"CUDA not available, using CPU backend. Error: {str(e)}")

        # Get output layer names
        ln = self.net.getLayerNames()
        output_layers_indices = self.net.getUnconnectedOutLayers()
        # Handle both formats (1D array or 2D array)
        if len(output_layers_indices.shape) > 1:
            # Handle case when indices are returned as nx1 array
            self.output_layers = [ln[i[0] - 1] for i in output_layers_indices]
        else:
            # Handle case when indices are returned as 1D array
            self.output_layers = [ln[i - 1] for i in output_layers_indices]

        logger.info("YOLO model loaded successfully")
        logger.info(
            f"Loaded {len(self.class_names)} classes: {', '.join(self.class_names[:5])}..."
        )
        logger.info(
            f"Confidence threshold: {self.conf_threshold}, NMS threshold: {self.nms_threshold}"
        )
        self.detection_results = []

    def run(self, img_arr):
        if img_arr is None:
            return img_arr, []

        # Clone the image for drawing
        image = img_arr.copy()
        height, width = image.shape[:2]

        # Define small font size, thickness and corner position
        font_size = 0.4
        thickness = 1
        x_pos = 5
        y_start = 15
        line_spacing = 15

        # Create a blob and pass it through the network
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)

        # Run inference
        start_time = time.time()
        outputs = self.net.forward(self.output_layers)
        inference_time = time.time() - start_time

        # Small text for inference time in top-left corner
        cv2.putText(
            image,
            f"Inf: {inference_time:.2f}s",
            (x_pos, y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 255, 0),
            thickness,
        )

        # Small text indicating detection is active
        cv2.putText(
            image,
            "Det Active",
            (x_pos, y_start + line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 0, 0),
            thickness,
        )

        # Process detections
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    # Scale bounding box coordinates back to original image
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, box_width, box_height) = box.astype("int")

                    # Get top-left coordinates of the box
                    x = int(center_x - (box_width / 2))
                    y = int(center_y - (box_height / 2))

                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Add small text for potential detections count
        cv2.putText(
            image,
            f"Pot: {len(boxes)}",
            (x_pos, y_start + 2 * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 0, 0),
            thickness,
        )

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )

        # Prepare detection results
        self.detection_results = []

        # Add small text for final detections count
        detection_count = len(indices) if len(indices) > 0 else 0
        cv2.putText(
            image,
            f"Det: {detection_count}",
            (x_pos, y_start + 3 * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 0, 0),
            thickness,
        )

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y, w, h) = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                class_name = self.class_names[class_id]

                # Store detection result
                self.detection_results.append(
                    {"class": class_name, "confidence": confidence, "box": (x, y, w, h)}
                )

                if self.draw_bboxes:
                    color = [int(c) for c in self.colors[class_id]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = f"{class_name}: {confidence:.2f}"
                    cv2.putText(
                        image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                    )
                    logger.info(
                        f"Detected {class_name} with confidence {confidence:.2f}"
                    )

        if self.detection_results:
            classes_found = set(d["class"] for d in self.detection_results)
            logger.info(f"Detected classes: {', '.join(classes_found)}")

        return image, self.detection_results

    def shutdown(self):
        pass
