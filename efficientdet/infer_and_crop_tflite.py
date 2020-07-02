import numpy as np
import tensorflow as tf
from absl import flags
from absl import app
import glob
import time
from PIL import Image
import os

flags.DEFINE_string('input_image', None, 'Input image path for inference.')
flags.DEFINE_string('output_image_dir', None, 'Output dir for inference.')

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

    image_files = glob.glob(FLAGS.input_image)
    os.makedirs(FLAGS.output_image_dir, exist_ok=True)

    total_exec_time = 0.
    for i, image_file in enumerate(image_files):
        pil_im = Image.open(image_file).convert("RGB")
        o_w, o_h = pil_im.size
        im = np.array(pil_im.resize((input_shape[2], input_shape[1])))
        im = np.expand_dims(im, axis=0)
        interpreter.set_tensor(input_details[0]['index'], im)
        start = time.time()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        exec_time = time.time() - start
        print(image_file, exec_time)
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
