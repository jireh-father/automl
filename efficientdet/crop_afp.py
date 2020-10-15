import argparse
import sys
import glob
import os
from pycocotools import mask
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument('--image_dir', dest='image_dir',
                        default='/home/irelin/Downloads/dog_eye_detection_dataset/03.images_coordinates', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='/home/irelin/resource/afp/entropion_binary_dataset/ori/0', type=str)
    parser.add_argument('--bbox_crop_margin_ratio', dest='bbox_crop_margin_ratio', default=0.3, type=float)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
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
        im = Image.open(image_file).convert("RGB")
        width, height = im.size
        rles = mask.frPyObjects(segmentation, height, width)
        rle = mask.merge(rles)
        bbox = mask.toBbox(rle)

        bbox[0] = max(round(bbox[0] - (bbox[2] * args.bbox_crop_margin_ratio)), 0)
        bbox[1] = max(round(bbox[1] - (bbox[3] * args.bbox_crop_margin_ratio)), 0)
        bbox[2] = min(round(bbox[2] + (bbox[2] * args.bbox_crop_margin_ratio)), width - bbox[0])
        bbox[3] = min(round(bbox[3] + (bbox[3] * args.bbox_crop_margin_ratio)), height - bbox[1])

        crop_im = im.crop((bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]))
        output_image_path = os.path.join(args.output_dir, os.path.basename(image_file))
        crop_im.save(output_image_path)

print("complete")
