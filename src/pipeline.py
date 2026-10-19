import cv2

VIOLATION_CLASSES =[
    "Without Helmet",

    "Without Mask",
    "Without Vest",
    "Without Ear Protectors"
]


def extract_detections(result, model):
    detections = []

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]

        x1, y1, x2,  y2 = box.xyxy[0]

        detections.append({
            "class": name,
            "confidence": conf,
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })

    return detections


def detect_violations(detections):
    violations = []

    for d in detections:
        if d["confidence"] < 0.3:
            continue

        if d["class"]  == "Without Helmet":
            if "No Helmet" not in violations:
                violations.append("No Helmet")

        elif d["class"] == "Without Safety Vest":
            if "No Safety Vest" not in violations:
                violations.append("No Safety Vest")

    return violations


def annotate_image(result, violations):
    img = result.plot()

    if len(violations) > 0:
        cv2.putText(
            img,
            "VIOLATION DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        y_offset = 80
        for v in violations:
            cv2.putText(
                img,
                v,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            y_offset += 30

    else:
        cv2.putText(
            img,
            "ALL SAFE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )


    return img