import json
import argparse
import sys
import glob
import os
import datetime
from pycocotools import mask
import imagesize


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument('--image_dir', dest='image_dir', default=None, type=str)
    parser.add_argument('--annotation_files', dest='annotation_files', default=None, type=str)
    parser.add_argument('--output_path', dest='output_path', default=None, type=str)
    # parser.add_argument('--output_image_dir', dest='output_image_dir', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def init_coco_annotation():
    coco_output = {}
    coco_output["info"] = {
        "description": "This is stable 1.0 version of the 2014 MS COCO dataset.",
        "url": "http://mscoco.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "Microsoft COCO group",
        "date_created": "2015-01-27 09:11:52.357475"
    }
    coco_output["type"] = "instances"
    coco_output["license"] = [{
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    }]
    coco_output["images"] = []

    coco_output["annotations"] = []
    coco_output["categories"] = [{
        "supercategory": "eye",
        "id": 1,
        "name": "eye"
    }, {
        "supercategory": "not_eye",
        "id": 2,
        "name": "not_eye"
    }, ]
    return coco_output


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # os.makedirs(args.output_image_dir, exist_ok=True)

    polygon_files = glob.glob(os.path.join(args.image_dir, "*.xy"))
    if args.annotation_files:
        annotation_files = args.annotation_files.split(",")
    else:
        annotation_files = None

    coco_output = init_coco_annotation()
    image_id_map = {}
    bbox_id_map = {}
    anno_sample = {"segmentation": [],
                   "area": 100.0,
                   "iscrowd": 0,
                   "image_id": 1,
                   "bbox": [1, 1, 1, 1],
                   "category_id": 1,
                   "id": 1}

    for i, polygon_file in enumerate(polygon_files):
        image_file = os.path.splitext(polygon_file)[0]
        if not os.path.isfile(image_file):
            print("not image file", image_file)
            continue

        segmentation = [
            [float(coord.strip()) for coord in open(polygon_file).readline().split(" ")[1:] if coord and coord.strip()]]
        width, height = imagesize.get(image_file)
        rles = mask.frPyObjects(segmentation, height, width)
        rle = mask.merge(rles)
        bbox = mask.toBbox(rle)
        area = mask.area(rle)

        image_fn = os.path.basename(image_file)

        if image_fn not in image_id_map:
            image_id_map[image_fn] = len(image_id_map) + 1

        coco_output["images"].append({
            "license": 1,
            "url": "",
            "file_name": image_fn,
            "height": height,
            "width": width,
            "date_captured": datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            "id": image_id_map[image_fn]
        })

        coco_output["annotations"].append({
            "segmentation": segmentation,
            "area": int(area),
            "iscrowd": 0,
            "image_id": i + 1,
            "bbox": list(bbox),
            "category_id": 1,
            "id": image_id_map[image_fn]
        })

    if annotation_files:
        for annotation_file in annotation_files.split(","):
            annotations = json.load(open(annotation_files))
            for image_fn in annotations:
                annotation = annotations[image_fn]
                if image_fn not in image_id_map:
                    image_id_map[image_fn] = len(image_id_map) + 1
                    coco_output["images"].append({
                        "license": 1,
                        "url": "",
                        "file_name": image_fn,
                        "height": annotation["height"],
                        "width": annotation["width"],
                        "date_captured": datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                        "id": image_id_map[image_fn]
                    })

                if image_fn not in bbox_id_map:
                    bbox_id_map[image_fn] = 0

                for bbox in annotation["bbox"]:
                    bbox_id_map[image_fn] += 1
                    coco_output["annotations"].append({
                        "segmentation": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y1"], bbox["x2"], bbox["y2"],
                                         bbox["x1"], bbox["y2"]],
                        "area": int((bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])),
                        "iscrowd": 0,
                        "image_id": image_id_map[image_fn],
                        "bbox": [bbox["x1"], bbox["y1"], bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"]],
                        "category_id": bbox["label"],
                        "id": bbox_id_map[image_fn]
                    })
    # {
    #   "0_0_0_Google_0546.jpeg": {
    #     "width": 698,
    #     "height": 663,
    #     "bbox": [
    #       {
    #         "x1": 416.49407958984375,
    #         "y1": 357.9506530761719,
    #         "x2": 685.0248413085938,
    #         "y2": 637.7289428710938,
    #         "label": 1
    #       }
    #     ]
    #   },
    #   "0_0_13_Google_0671.jpeg": {
    #     "width": 1200,
    #     "height": 560,
    #     "bbox": [
    #       {
    #         "x1": 557.7907104492188,
    #         "y1": 283.9653015136719,
    #         "x2": 713.557373046875,
    #         "y2": 401.5119934082031,
    #         "label": 1
    #       }
    #     ]
    #   }
    # }

    json.dump(coco_output, open(args.output_path, "w+"))
    print("complete")
