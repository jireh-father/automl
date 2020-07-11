from absl import flags
from absl import app
import glob
import os
import shutil

flags.DEFINE_string('image_dir', None, 'Input image path for inference.')
flags.DEFINE_string('output_image_dir', None, 'Output dir for inference.')

flags.DEFINE_float('target_nums_image', 6697, 'Score threshold to show box.')

FLAGS = flags.FLAGS


def main(_):
    image_dirs = glob.glob(os.path.join(FLAGS.image_dir, "*"))

    target_nums_image = FLAGS.target_nums_image
    assert target_nums_image > 0
    assert len(image_dirs) > 0

    print("start")
    for image_dir in image_dirs:
        print("image_dir")
        image_files = glob.glob(os.path.join(image_dir, "*"))
        output_dir = os.path.join(FLAGS.output_image_dir, os.path.basename(image_dir))
        os.makedirs(output_dir, exist_ok=True)
        if len(image_files) >= target_nums_image:
            for image_file in image_files:
                shutil.copy(image_file, output_dir)
            print("copy completed not oversampled")
        else:
            cp_cnt = target_nums_image - len(image_files)
            while True:
                for image_file in image_files:
                    if cp_cnt < 1:
                        break
                    shutil.copy(image_file, output_dir)
                    cp_cnt -= 1
                if cp_cnt < 1:
                    break
            print("copy completed", cp_cnt)
    print("complete")


if __name__ == '__main__':
    # tf.disable_eager_execution()
    app.run(main)
