import numpy as np
import tensorflow as tf
from absl import flags
from absl import app
import glob
import time
from PIL import Image
import os
import json

flags.DEFINE_string('input_image', None, 'Input image path for inference.')
flags.DEFINE_string('output_path', None, 'Output dir for inference.')
flags.DEFINE_string('real_image_dir', None, 'Output dir for inference.')

flags.DEFINE_float('min_score_thresh', 0.3, 'Score threshold to show box.')
flags.DEFINE_integer('target_label_idx', 1, 'Score threshold to show box.')

flags.DEFINE_string('tflite_path', None, 'Path for exporting tflite file.')

FLAGS = flags.FLAGS


def main(_):
    model_path = FLAGS.tflite_path
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    image_file_list = glob.glob(FLAGS.input_image)
    image_file_list.sort()
    if os.path.dirname(FLAGS.output_path):
        os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)
    real_image_dict = {}
    label_dict = {}
    for image_file in image_file_list:
        splitext = os.path.splitext(image_file)
        ext = splitext[1]
        fp = splitext[0]
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
        if label_dir not in label_dict:
            cur_label = len(label_dict) + 2
            label_dict[label_dir] = cur_label
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
        im = np.array(pil_im.resize((input_shape[2], input_shape[1])))
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
        eye_indexes = np.squeeze(np.argwhere(output_data[0, :, 6] == FLAGS.target_label_idx), 1)
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
            print(eyes_bboxes)
            if eyes_bboxes:
                for j, bbox_idx in enumerate(bbox_idxs):
                    if bbox_idx >= len(eyes_bboxes):
                        raise Exception("invalid bbox index!", bbox_idx, eyes_bboxes)
                    bbox = eyes_bboxes[bbox_idx]
                    bbox["label"] = labels[j]
                    annotations[image_fn]["bbox"].append(bbox)
        else:
            print("no eyes")

    json.dump(annotations, open(FLAGS.output_path, "w+"))
    print("avg exec time", total_exec_time / len(real_image_dict), "total images", len(real_image_dict))


if __name__ == '__main__':
    # tf.disable_eager_execution()
    app.run(main)
