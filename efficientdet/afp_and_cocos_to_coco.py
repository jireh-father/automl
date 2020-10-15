import json
import argparse
import sys
import glob
import os
import datetime
from pycocotools import mask
import imagesize
import random
import shutil
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument('--afp_image_dir', dest='afp_image_dir', default=None, type=str)
    parser.add_argument('--custom_image_dirs', dest='custom_image_dir', default=None, type=str)
    parser.add_argument('--coco_anno_files_pattern', dest='coco_anno_files_pattern', default=None, type=str)
    parser.add_argument('--output_dir', dest='output_dir', default=None, type=str)
    parser.add_argument('--output_image_dir', dest='output_image_dir', default=None, type=str)
    parser.add_argument('--vis_dir', dest='vis_dir', default=None, type=str)

    parser.add_argument('--label_dir', dest='label_dir', default=None, type=str)

    parser.add_argument('--start_idx', dest='start_idx', default=1, type=int)
    parser.add_argument('--seed', dest='seed', default=1, type=int)
    parser.add_argument('--afp_ratio', dest='afp_ratio', default=0.15, type=float)
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


def get_afp_anno_list(polygon_files, output_image_dir, vis_dir):
    anno_data = []

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

        if output_image_dir:
            shutil.copy(image_file, output_image_dir)

        if vis_dir:
            im = Image.open(image_file).convert("RGB")
            draw = ImageDraw.Draw(im)
            draw.rectangle(xy=[bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]], fill=(255, 0, 0, 100),
                           outline='red')

            im.save(os.path.join(vis_dir, "afp_{}.jpg".format(i)))

        anno_data.append({
            'images': {
                "license": 1,
                "url": "",
                "file_name": image_fn,
                "height": height,
                "width": width,
                "date_captured": datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            },
            'annotations': [{
                "segmentation": segmentation,
                "area": int(area),
                "iscrowd": 0,
                "bbox": list(bbox),
                "category_id": 1,
            }]
        })

    return anno_data


def get_custom_anno_list(anno_files, image_dir, output_image_dir, vis_dir):
    anno_list = []

    for anno_file in anno_files:
        anno_dict = json.load(open(anno_file))
        tmp_anno_dict = {}
        for anno_item in anno_dict['annotations']:
            if anno_item['image_id'] not in tmp_anno_dict:
                tmp_anno_dict[anno_item['image_id']] = {}
            if 'annotations' not in tmp_anno_dict[anno_item['image_id']]:
                tmp_anno_dict[anno_item['image_id']]['annotations'] = []
            tmp_anno_dict[anno_item['image_id']]['annotations'].append(anno_item)

        for image_item in anno_dict['images']:
            if image_item['id'] not in tmp_anno_dict:
                continue
            tmp_anno_dict[image_item['id']]['images'] = image_item
            if output_image_dir:
                image_path = os.path.join(image_dir, image_item['file_name'])
                shutil.copy(image_path, output_image_dir)

        anno_list += list(tmp_anno_dict.values())

    if vis_dir:
        vis_image_id = 1
        for anno_dict in anno_list:
            image_item = anno_dict['images']
            image_path = os.path.join(image_dir, image_item['file_name'])
            im = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(im)
            for anno_item in anno_dict['annotations']:
                bbox = anno_item['bbox']
                draw.rectangle(xy=[bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]], fill=(255, 0, 0, 100),
                               outline='red')
            im.save(os.path.join(vis_dir, "custom_{}.jpg".format(vis_image_id)))
            vis_image_id += 1

    return anno_list


def get_coco(anno_list):
    coco_output = init_coco_annotation(args.label_dir, args.start_idx)
    cur_anno_id = 1

    for i, anno_item in enumerate(anno_list):
        image_id = i + 1
        anno_item['images']['id'] = image_id
        coco_output["images"].append(anno_item['images'])

        for tmp_anno in anno_item['annotations']:
            tmp_anno['image_id'] = image_id
            tmp_anno['id'] = cur_anno_id
            cur_anno_id += 1
            coco_output["annotations"].append(tmp_anno)
    return coco_output


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    if args.output_image_dir:
        os.makedirs(args.output_image_dir, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    anno_sample = {"segmentation": [],
                   "area": 100.0,
                   "iscrowd": 0,
                   "image_id": 1,
                   "bbox": [1, 1, 1, 1],
                   "category_id": 1,
                   "id": 1}

    seg_files = []
    if args.afp_image_dir:
        seg_files = glob.glob(os.path.join(args.afp_image_dir, "*.xy"))
    random.shuffle(seg_files)
    seg_files = seg_files[:int(len(seg_files) * args.afp_ratio)]

    afp_anno_list = get_afp_anno_list(seg_files, args.output_image_dir, args.vis_dir)

    coco_anno_files = glob.glob(args.coco_anno_files_pattern)
    custom_anno_list = get_custom_anno_list(coco_anno_files, args.custom_image_dir, args.output_image_dir, args.vis_dir)

    random.shuffle(afp_anno_list)
    random.shuffle(custom_anno_list)

    afp_train_last_index = round(len(afp_anno_list) * args.train_ratio)
    train_afp_anno_list = afp_anno_list[:afp_train_last_index]
    val_afp_anno_list = afp_anno_list[afp_train_last_index:]

    custom_train_last_index = round(len(custom_anno_list) * args.train_ratio)
    train_custom_anno_list = custom_anno_list[:custom_train_last_index]
    val_custom_anno_list = custom_anno_list[custom_train_last_index:]

    train_anno_list = train_afp_anno_list + train_custom_anno_list

    train_coco_output = get_coco(train_anno_list)
    val_afp_coco_output = get_coco(val_afp_anno_list)
    val_custom_coco_output = get_coco(val_custom_anno_list)

    json.dump(train_coco_output, open(os.path.join(args.output_dir, "train_coco.json"), "w+"))
    json.dump(val_afp_coco_output, open(os.path.join(args.output_dir, "val_afp_coco.json"), "w+"))
    json.dump(val_custom_coco_output, open(os.path.join(args.output_dir, "val_custom_coco.json"), "w+"))
    print("complete")
