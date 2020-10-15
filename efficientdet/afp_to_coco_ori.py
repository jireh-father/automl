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
    parser.add_argument('--output_path', dest='output_path', default=None, type=str)

    parser.add_argument('--label_dir', dest='label_dir', default=None, type=str)

    parser.add_argument('--start_idx', dest='start_idx', default=1, type=int)
    parser.add_argument('--train_ratio', dest='train_ratio', default=0.9, type=float)
    # parser.add_argument('--output_image_dir', dest='output_image_dir', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def init_coco_annotation(label_dir, start_idx):
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

    if label_dir is None:
        coco_output["categories"] = [{
            "supercategory": "eye",
            "id": 1,
            "name": "eye"
        }, ]
    else:
        label_dirs = glob.glob(os.path.join(label_dir, "*"))
        label_dirs.sort()

        coco_output["categories"] = [
            {"supercategory": os.path.basename(dname), "id": i + start_idx, "name": os.path.basename(dname)} for
            i, dname
            in enumerate(label_dirs)]
    return coco_output


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # os.makedirs(args.output_image_dir, exist_ok=True)
    coco_output = init_coco_annotation(args.label_dir, args.start_idx)
    image_id_map = {}
    bbox_id_map = {}
    anno_sample = {"segmentation": [],
                   "area": 100.0,
                   "iscrowd": 0,
                   "image_id": 1,
                   "bbox": [1, 1, 1, 1],
                   "category_id": 1,
                   "id": 1}

    polygon_files = []
    if args.image_dir:
        polygon_files = glob.glob(os.path.join(args.image_dir, "*.xy"))
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

    json.dump(coco_output, open(args.output_path, "w+"))
    print("complete")
