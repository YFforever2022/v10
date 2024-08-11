import warnings
warnings.filterwarnings("ignore", message='.*__floordiv__ is deprecated.*')

from ultralytics import YOLOv10
import multiprocessing
import argparse

def parse_args():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train、predict、val')
    parser.add_argument('--weights', type=str, default='./1/yolov10n.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='./1/datasets/test/data.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch', type=int, default=1, help='Sets the number of images per batch. Use -1 for AutoBatch, which automatically adjusts based on GPU memory availability.')
    parser.add_argument('--imgsz', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--resume', type=int, default=0, help='resume most recent training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--patience', type=int, default=0, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--source', type=str, default='./1/datasets/test/images', help='images path')
    parser.add_argument('--iou', type=float, default=0.6, help='Sets the Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Helps in reducing duplicate detections.')
    parser.add_argument('--plots', type=int, default=1, help='When set to True, generates and saves plots of predictions versus ground truth for visual evaluation of the models performance.')
    parser.add_argument('--conf', type=float, default=0.25, help='Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.')
    parser.add_argument('--save', type=int, default=1, help='Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results.')
    parser.add_argument('--save_txt', type=int, default=1, help='Saves detection results in a text file, following the format [class] [x_center] [y_center] [width] [height] [confidence]. Useful for integration with other analysis tools.')
    parser.add_argument('--show_labels', type=int, default=1, help='Displays labels for each detection in the visual output. Provides immediate understanding of detected objects.')
    parser.add_argument('--show_conf', type=int, default=1, help='Displays the confidence score for each detection alongside the label. Gives insight into the models certainty for each detection.')
    parser.add_argument('--show_boxes', type=int, default=1, help='Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames.')
    parser.add_argument('--line_width', type=int, default=None, help='Specifies the line width of bounding boxes. If None, the line width is automatically adjusted based on the image size. Provides visual customization for clarity.')
    parser.add_argument('--cache', type=int, default=1, help='1=ram 0=disk')
    return parser.parse_args()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    args = parse_args()
    args.resume = True if args.resume == 1 else False
    args.plots = True if args.plots == 1 else False
    args.save = True if args.save == 1 else False
    args.save_txt = True if args.save_txt == 1 else False
    args.show_labels = True if args.show_labels == 1 else False
    args.show_conf = True if args.show_conf == 1 else False
    args.show_boxes = True if args.show_boxes == 1 else False
    args.cache = True if args.cache == 1 else False

    if args.mode == 'train':
        model = YOLOv10(args.weights)
        results = model.train(data=args.data,
                              batch=args.batch,
                              epochs=args.epochs,
                              imgsz=args.imgsz,
                              resume=args.resume,
                              device=args.device,
                              workers=0,
                              cache=args.cache,
                              amp=False,
                              project=args.project,
                              name=args.name,
                              patience=args.patience)
    elif args.mode == 'predict':
        model = YOLOv10(args.weights)
        results = model.predict(source=args.source,
                                imgsz=args.imgsz,
                                conf=args.conf,
                                save=args.save,
                                save_txt=args.save_txt,
                                show_labels=args.show_labels,
                                show_conf=args.show_conf,
                                show_boxes=args.show_boxes,
                                line_width=args.line_width,
                                workers=0,
                                device=args.device,
                                project=args.project,
                                name=args.name)
        for r in results:
            int_cls = r.boxes.cls.int().tolist()
            float_conf = r.boxes.conf.float().tolist()
            int_xyxy = r.boxes.xyxy.int().tolist()
            for index, box in enumerate(int_xyxy):
                x = box[0]
                y = box[1]
                w = abs(box[2] - box[0])
                h = abs(box[3] - box[1])
                cls = int_cls[index]
                name = model.names[cls]
                conf = "{:.2f}".format(float_conf[index])
                ret = str(cls) + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',' + str(conf) + ',' + str(name)
                print(ret)
            print()
    elif args.mode == 'val':
        model = YOLOv10(args.weights)
        results = model.val(data=args.data,
                            imgsz=args.imgsz,
                            batch=args.batch,
                            conf=args.conf,
                            save=args.save,
                            device=args.device,
                            project=args.project,
                            name=args.name,
                            workers=0)
