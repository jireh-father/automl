import numpy as np
import tensorflow as tf
from absl import flags
from absl import app
import glob
import time
from PIL import Image
import os
import shutil
import json
import random

flags.DEFINE_string('input_image', None, 'Input image path for inference.')
flags.DEFINE_string('label_dir', None, 'Input image path for inference.')
flags.DEFINE_string('output_path', None, 'Output dir for inference.')
flags.DEFINE_string('output_image_dir', None, 'Output dir for inference.')
flags.DEFINE_string('vis_image_dir', None, 'Output dir for inference.')
flags.DEFINE_string('real_image_dir', None, 'Output dir for inference.')

flags.DEFINE_float('min_score_thresh', 0.3, 'Score threshold to show box.')
flags.DEFINE_integer('target_label_idx', None, 'Score threshold to show box.')

flags.DEFINE_string('tflite_path', None, 'Path for exporting tflite file.')

flags.DEFINE_integer('start_index', 1, 'Path for exporting tflite file.')
flags.DEFINE_integer('random_seed', 1, 'Path for exporting tflite file.')

flags.DEFINE_boolean('use_bbox_aug', False, 'Path for exporting tflite file.')
flags.DEFINE_float('bbox_aug_ratio', 0.1, 'Path for exporting tflite file.')

FLAGS = flags.FLAGS

def resize_and_crop_image(img, output_size):
    height, width = img.shape[:2]

    scale = output_size / float(max(width, height))

    if scale != 1.0:
        height = int(round(height * scale))
        width = int(round(width * scale))
        interpolation = cv2.INTER_LINEAR
        img = cv2.resize(img, (width, height), interpolation=interpolation)

    img = cv2.copyMakeBorder(img, 0, output_size - height, 0, output_size - width, cv2.BORDER_CONSTANT, value=0)

    return img, width, height

def main(_):
    model_path = FLAGS.tflite_path
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    if FLAGS.random_seed:
        random.seed(FLAGS.random_seed)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    image_file_list = glob.glob(FLAGS.input_image)
    label_dirs = glob.glob(os.path.join(FLAGS.label_dir, "*"))
    label_dirs.sort()
    label_dict = {os.path.basename(dname): i + FLAGS.start_index for i, dname in enumerate(label_dirs)}
    image_file_list.sort()
    if os.path.dirname(FLAGS.output_path):
        os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)
    os.makedirs(FLAGS.output_image_dir, exist_ok=True)
    os.makedirs(FLAGS.vis_image_dir, exist_ok=True)
    real_image_dict = {}
    for image_file in image_file_list:
        splitext = os.path.splitext(image_file)
        ext = splitext[1]
        fp = splitext[0]

        real_file_name = "_".join(os.path.basename(fp).split("_")[1:-1])
        searched = glob.glob(os.path.join(FLAGS.real_image_dir, real_file_name + ".*"))
        if len(searched) == 1:
            real_file_path = searched[0]
        else:
            real_file_name = "_".join(os.path.basename(fp).split("_")[:-1])
            real_file_path = os.path.join(FLAGS.real_image_dir, real_file_name + ext)
            if not os.path.isfile(real_file_path):
                real_file_path = os.path.join(FLAGS.real_image_dir, real_file_name + ".jpeg")
            if not os.path.isfile(real_file_path):
                real_file_path = os.path.join(FLAGS.real_image_dir, real_file_name + ".png")
            if not os.path.isfile(real_file_path):
                real_file_path = os.path.join(FLAGS.real_image_dir, real_file_name + ".bmp")
        if real_file_path not in real_image_dict:
            real_image_dict[real_file_path] = []
        bbox_idx = int(os.path.basename(fp).split("_")[-1])
        label_dir = os.path.basename(os.path.dirname(image_file))
        real_image_dict[real_file_path].append([bbox_idx, label_dict[label_dir]])

    annotations = {}
    total_exec_time = 0.
    for i, image_file in enumerate(real_image_dict):
        bbox_idx_and_labels = real_image_dict[image_file]
        print(image_file, bbox_idx_and_labels)
        bbox_idxs = [item[0] for item in bbox_idx_and_labels]
        labels = [item[1] for item in bbox_idx_and_labels]
        pil_im = Image.open(image_file).convert("RGB")
        o_w, o_h = pil_im.size
        image_fn = os.path.basename(image_file)
        annotations[image_fn] = {"width": o_w, "height": o_h, "bbox": []}
        # im = np.array(pil_im.resize((input_shape[2], input_shape[1])))
        im = np.array(pil_im)
        # im = normalize_image(np.array(pil_im))
        im, r_w, r_h = resize_and_crop_image(im, input_shape[1])
        im = np.expand_dims(im, axis=0)
        interpreter.set_tensor(input_details[0]['index'], im)
        start = time.time()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        exec_time = time.time() - start
        print(i, len(real_image_dict), image_file, exec_time)
        total_exec_time += exec_time
        r_h = input_shape[1]
        r_w = input_shape[2]
        if FLAGS.target_label_idx is not None:
            eye_indexes = np.squeeze(np.argwhere(output_data[0, :, 6] == FLAGS.target_label_idx), 1)
            eye_indexes = np.squeeze(np.argwhere(output_data[0][eye_indexes][:, 5] > 0.), 1)
        else:
            eye_indexes = np.squeeze(np.argwhere(output_data[0, :, 5] > 0.), 1)
        if len(eye_indexes) > 0:
            top_k = 30
            top_k_indexes = output_data[0][eye_indexes][:, 5].argsort()[::-1][:top_k]
            scores = output_data[0][top_k_indexes][:, 5]
            bboxes = output_data[0][top_k_indexes][:, 1:5]
            eyes_bboxes = []
            for j, score in enumerate(scores):
                if score < FLAGS.min_score_thresh:
                    print("skip", score, "<", FLAGS.min_score_thresh)
                    break
                r_y1, r_x1, r_y2, r_x2 = bboxes[j]

                y1 = r_y1 / r_h * o_h
                y2 = r_y2 / r_h * o_h
                x1 = r_x1 / r_w * o_w
                x2 = r_x2 / r_w * o_w
                print(o_w, o_h, (x1, y1, x2, y2))
                eyes_bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

                crop_im = pil_im.crop((x1, y1, x2, y2))
                output_filename = "{}_{}.jpg".format(os.path.splitext(os.path.basename(image_file))[0], j)
                output_path = os.path.join(FLAGS.vis_image_dir, output_filename)
                crop_im.save(output_path)

            print(eyes_bboxes)
            if eyes_bboxes:
                if not os.path.isfile(os.path.join(FLAGS.output_image_dir, os.path.basename(image_file))):
                    shutil.copy(image_file, FLAGS.output_image_dir)
                dup_idx = False
                if len(set(bbox_idxs)) < len(bbox_idxs):
                    dup_idx = True
                bbox_idx_map = {}
                for j, bbox_idx in enumerate(bbox_idxs):
                    if bbox_idx >= len(eyes_bboxes):
                        raise Exception("invalid bbox index!", bbox_idx, eyes_bboxes)
                    bbox = eyes_bboxes[bbox_idx]
                    bbox["label"] = labels[j]
                    if dup_idx and FLAGS.use_bbox_aug and bbox_idx in bbox_idx_map:
                        w = bbox["x2"] - bbox["x1"]
                        h = bbox["y2"] - bbox["y1"]
                        w_aug_max = round(w * FLAGS.bbox_aug_ratio)
                        h_aug_max = round(h * FLAGS.bbox_aug_ratio)
                        x1_aug = random.randint(0, w_aug_max) - (w_aug_max // 2)
                        x2_aug = random.randint(0, w_aug_max) - (w_aug_max // 2)
                        y1_aug = random.randint(0, h_aug_max) - (h_aug_max // 2)
                        y2_aug = random.randint(0, h_aug_max) - (h_aug_max // 2)
                        bbox["x1"] += x1_aug
                        bbox["x2"] += x2_aug
                        bbox["y1"] += y1_aug
                        bbox["y2"] += y2_aug
                    bbox_idx_map[bbox_idx] = True

                    annotations[image_fn]["bbox"].append(bbox)
        else:
            print("no eyes")

    json.dump(annotations, open(FLAGS.output_path, "w+"))
    print("avg exec time", total_exec_time / len(real_image_dict), "total images", len(real_image_dict))


if __name__ == '__main__':
    # tf.disable_eager_execution()
    app.run(main)
