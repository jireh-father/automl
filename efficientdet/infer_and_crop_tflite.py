import numpy as np
import tensorflow as tf
from absl import flags
from absl import app
import glob
import time
from PIL import Image
import os
import cv2

flags.DEFINE_string('input_image', None, 'Input image path for inference.')
flags.DEFINE_string('output_image_dir', None, 'Output dir for inference.')

flags.DEFINE_float('min_score_thresh', 0.3, 'Score threshold to show box.')
flags.DEFINE_integer('target_label_idx', 1, 'Score threshold to show box.')

flags.DEFINE_string('tflite_path', None, 'Path for exporting tflite file.')

FLAGS = flags.FLAGS


def normalize_image(image):
    """Normalize the image to zero mean and unit variance."""
    # The image normalization is identical to Cloud TPU ResNet.
    image = image.astype(np.float32) / 255.

    offset = np.array([0.485, 0.456, 0.406])
    offset = np.expand_dims(offset, axis=0)
    offset = np.expand_dims(offset, axis=0)
    image -= offset

    scale = np.array([0.229, 0.224, 0.225])
    scale = np.expand_dims(scale, axis=0)
    scale = np.expand_dims(scale, axis=0)
    image /= scale

    return image


def resize_and_crop_image(img, output_size):
    height, width = img.shape[:2]

    scale = output_size / float(max(width, height))

    if scale != 1.0:
        height = int(round(height * scale))
        width = int(round(width * scale))
        interpolation = cv2.INTER_LINEAR
        img = cv2.resize(img, (width, height), interpolation=interpolation)

    img = cv2.copyMakeBorder(img, 0, output_size - height, 0, output_size - width, cv2.BORDER_CONSTANT, value=0)

    return img


def main(_):
    model_path = FLAGS.tflite_path
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    image_files = glob.glob(FLAGS.input_image)
    os.makedirs(FLAGS.output_image_dir, exist_ok=True)

    total_exec_time = 0.
    for i, image_file in enumerate(image_files):
        pil_im = Image.open(image_file).convert("RGB")
        o_w, o_h = pil_im.size
        im = np.array(pil_im)
        # im = normalize_image(np.array(pil_im))
        im = resize_and_crop_image(im, input_shape[1])

        # im = np.array(pil_im.resize((input_shape[2], input_shape[1])))
        im = np.expand_dims(im, axis=0)
        interpreter.set_tensor(input_details[0]['index'], im)
        start = time.time()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        exec_time = time.time() - start
        print(i, len(image_files), image_file, exec_time)
        total_exec_time += exec_time
        r_h = input_shape[1]
        r_w = input_shape[2]
        eye_indexes = np.squeeze(np.argwhere(output_data[0, :, 6] == FLAGS.target_label_idx), 1)
        eyes = []
        if len(eye_indexes) > 0:
            top_k = 30
            top_k_indexes = output_data[0][eye_indexes][:, 5].argsort()[::-1][:top_k]
            scores = output_data[0][top_k_indexes][:, 5]
            bboxes = output_data[0][top_k_indexes][:, 1:5]
            for j, score in enumerate(scores):
                if score < FLAGS.min_score_thresh:
                    print("skip", score, "<", FLAGS.min_score_thresh)
                    break
                r_y1, r_x1, r_y2, r_x2 = bboxes[j]

                y1 = r_y1 / r_h * o_h
                y2 = r_y2 / r_h * o_h
                x1 = r_x1 / r_w * o_w
                x2 = r_x2 / r_w * o_w
                if x2 - x1 < 1 or y2 - y1 < 1:
                    print("small box side")
                    continue

                print(o_w, o_h, (x1, y1, x2, y2))
                crop_im = pil_im.crop((x1, y1, x2, y2))
                output_filename = "{}_{}.jpg".format(os.path.splitext(os.path.basename(image_file))[0], j)
                output_path = os.path.join(FLAGS.output_image_dir, output_filename)
                crop_im.save(output_path)
                # eyes.append([x1, y1, x2, y2])
            print(eyes)
        else:
            print("no eyes")
    print("avg exec time", total_exec_time / len(image_files), "total images", len(image_files))


if __name__ == '__main__':
    # tf.disable_eager_execution()
    app.run(main)
