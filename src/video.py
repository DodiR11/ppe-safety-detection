import cv2
import numpy as np
from collections import deque

from src.pipeline import extract_detections, detect_violations, annotate_image


def process_video(model, input_path, output_path, conf=0.4):
    history =    deque(maxlen=5)

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc= cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = 0
    violation_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results =model.predict(source=frame, conf=conf, verbose=False)

        for r in results:
            detections= extract_detections(r, model)
            violations = detect_violations(detections)

            current = 1 if violations else 0
            history.append(current)

            stable_violation = sum(history) >= 3

            if stable_violation:
                violation_frames += 1
                annotated_frame = annotate_image(r, violations)
            else:
                annotated_frame = annotate_image(r, [])

            annotated_frame= np.array(annotated_frame).astype("uint8")

            out.write(annotated_frame)

        total_frames += 1

    cap.release()
    out.release()

    return {
        "total_frames": total_frames,
        "violation_frames": violation_frames
    }