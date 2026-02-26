import cv2
import torch
from ultralytics import YOLO

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

model = YOLO(r"YOLO-etrikes.pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 80: "etrike"}
logged_ids = {cls: set() for cls in vehicle_classes}

results = model.track(
    source=r"path_to_video.mp4",
    stream=True,
    show=True,
    tracker="bytetrack.yaml",
    half=True if device.startswith("cuda") else False
)

for result in results:

    # Draw boxes and tracking info on the frame
    frame = result.plot()
    resized_frame = cv2.resize(frame, (1600, 900))

    boxes = result.boxes
    if boxes.id is not None:
        for box_id, cls in zip(boxes.id.cpu().numpy(), boxes.cls.cpu().numpy()):
            box_id = int(box_id)
            cls = int(cls)

            if cls in vehicle_classes:
                if box_id not in logged_ids[cls]:
                    print(f"Logged new {vehicle_classes[cls]} with ID: {box_id}")
                    logged_ids[cls].add(box_id)

cv2.destroyAllWindows()


def plural(name):
    if name == "bus":
        return "buses"
    else:
        return "{0}s".format(name)


for cls, name in vehicle_classes.items():
    print(f" Total unique {plural(name)} detected: {len(logged_ids[cls])}")

