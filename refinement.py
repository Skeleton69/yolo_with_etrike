from ultralytics import YOLO
import numpy as np

def main():

    model = YOLO("C:/Users/goper/YOLO_etrikes/experiments/yolo_etrikes/weights/best.pt")

    model.train(
        data="dataset/coco-with-etrikes.yaml",
        imgsz=512,
        epochs=25,
        batch=2,
        lr0=0.000005,
        lrf=0.0001,
        warmup_epochs=2,
        warmup_momentum=0.8,
        optimizer="AdamW",
        patience=10,
        amp=True,
        augment=True,
        mosaic=0.3,
        mixup=0.0,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        project="experiments",
        name="refinement",
        workers=4,
        freeze=15,
        device=0,
        weight_decay=0.00005,
        close_mosaic=10,
        box=8.5,
    )

    # Validation
    results = model.val(data="dataset/coco-with-etrikes.yaml", workers=0)
    box = results.box

    # Overall Metrics
    print("=" * 60)
    print("REFINEMENT RESULTS")
    print("=" * 60)
    print("Precision:", box.mp)
    print("Recall:", box.mr)
    print("mAP@0.5:", box.map50)
    print("mAP@0.5:0.95:", box.map)

    # Etrike only Metrics
    print("\n" + "=" * 60)
    print("E-TRIKE METRICS (Class 80)")
    print("=" * 60)
    print("Precision:", box.p[80])
    print("Recall:", box.r[80])
    print("F1:", box.f1[80])
    print("mAP@0.5:", box.ap50[80])
    print("mAP@0.5:0.95:", box.ap[80])

    # COCO-only metrics
    coco_ids = list(range(80))
    print("\n" + "=" * 60)
    print("COCO-ONLY METRICS (Classes 0–79)")
    print("=" * 60)
    coco_precision = np.nanmean(box.p[coco_ids])
    coco_recall = np.nanmean(box.r[coco_ids])
    coco_f1 = np.nanmean(box.f1[coco_ids])
    coco_map50 = np.nanmean(box.ap50[coco_ids])
    coco_map = np.nanmean(box.ap[coco_ids])

    print("Precision:", coco_precision)
    print("Recall:", coco_recall)
    print("F1:", coco_f1)
    print("mAP@0.5:", coco_map50)
    print("mAP@0.5:0.95:", coco_map)


if __name__ == "__main__":
    main()