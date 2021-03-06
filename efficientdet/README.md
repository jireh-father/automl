# EfficientDet

[1] Mingxing Tan, Ruoming Pang, Quoc V. Le. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020.
Arxiv link: https://arxiv.org/abs/1911.09070

Updates:

  - **Apr22: Speed up end-to-end latency: D0 has up to >200 FPS throughput on Tesla V100.**
    * A great collaboration with [@fsx950223](https://github.com/fsx950223).
  - Apr1: Updated results for test-dev and added EfficientDet-D7.
  - Mar26: Fixed a few bugs and updated all checkpoints/results.
  - Mar24: Added tutorial with visualization and coco eval.
  - Mar 13: Released the initial code and models.

**Quick start tutorial: [tutorial.ipynb](tutorial.ipynb)**

**Quick install dependencies: ```pip install -r requirements.txt```**

## 0. Training AFP Dataset
### AI For Pet 데이터셋 구조
한 이미지에 강아지 눈 bbox는 한개만 존재함.
이미지 파일명.xy파일에 눈의 segmentation 좌표정보가 있다.
- 14_001.jpg.xy
- 14_001.jpg
- 14_002.jpg.xy
- 14_002.jpg
- ....

### 서버 접속 및 가상환경 활성화
1. 서버접속
2. source t2/bin/activate
3. cd source/automl/efficientdet

### a. AI For Pet 데이터셋 coco annotation 형식으로 변경
```bash
python afp_to_coco.py --image_dir=[ai for pet 데이터셋 경로] \
--output_path=[coco annotation파일 경로, ex: afp_coco.json]
```

### b. coco 데이터셋 tflite 형태로 변경
#### 학습 데이터셋
```bash
python -m dataset.create_coco_tfrecord --image_dir=[ai for pet 데이터셋 경로] \
--object_annotations_file=[학습용 coco annotation파일 경로] \
--output_file_prefix=[tflite 출력 경로. ex) ./tflite/train]
```

#### validation 데이터셋
```bash
python -m dataset.create_coco_tfrecord --image_dir=[ai for pet 데이터셋 경로] \
--object_annotations_file=[validation의 coco annotation파일 경로] \
--output_file_prefix=[tflite 출력 경로. ex) ./tflite/val]
```

### c. 학습

#### Create a config file for the PASCAL VOC dataset called voc_config.yaml and put this in it.

      num_classes: 2
      var_freeze_expr: '(efficientnet|fpn_cells|resample_p6)'
      label_id_mapping: {0: background, 1: eye}
      
hparams_config.py 파일 default_detection_configs 함수를 참고하여
추가적인 하이퍼파라미터를 세팅할 수 있다.

#### Download a pretrained model
아래 "2. Pretrained EfficientDet Checkpoints" 에서 원하는 모델 다운로드

#### training
```bash
python -u main.py --mode=train_and_eval \
--training_file_pattern=[train tfrecord디렉토리/train*.tfrecord] \
--validation_file_pattern=[validation tfrecord디렉토리/val*.tfrecord] \
--model_name=efficientdet-d0(기타 d1~d7 사용가능) --model_dir=[저장할 모델 디렉토리] \
--ckpt=[다운로드한 pretrained 모델 경로, ex) ./pretrained/efficientdet-d0] \
--train_batch_size=64 --eval_batch_size=64 --eval_samples=[eval이미지 갯수] \
--num_examples_per_epoch=[학습이미지 갯수] \
--num_epochs=50 --hparams=voc_config.yaml
```

### d. inference and crop bboxes by ckpt model
```bash
python -u model_inspect.py --runmode=infer_and_crop --model_name=efficientdet-d0(기타 d1~d7 사용가능) \
--hparams="num_classes=2" --max_boxes_to_draw=5 --min_score_thresh=[crop할 최소 score threshold, ex: 0.6] \
--ckpt_path=[학습된 ckpt 모델 디렉토리 혹은 파일명] \
--input_image=[테스트 이미지 경로. ex) test_image/* ] \
--output_image_dir=[crop된 이미지 저장 디렉토리] --batch_size=256
```

### convert ckpt model to saved model
```bash
python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 --ckpt_path=[ckpt모델 파일명] \
--hparams="num_classes=2" --saved_model_dir=[변경할 모델을 저장할 디렉토리]
```

### inference and crop bboxes by saved model
```bash
python model_inspect.py --runmode=saved_model_infer --model_name=efficientdet-d0 \
--saved_model_dir=[변경한 모델을 저장한 디렉토리] --input_image=[입력 이미지 경로, ex) test/*] \
--output_image_dir=[출력결과 디렉토리] --hparams="num_classes=2" --min_score_thresh=[threshold, ex) 0.5]
```

### e. convert ckpt model to tflite model
```bash
python model_inspect.py \
--runmode=saved_model \
--model_name=efficientdet-d0 \
--ckpt_path=[trained_model_ckpt_dir] \
--saved_model_dir=[saved_model_dir] \
--tflite_path=[tflite_output_path] \
--hparams="num_classes=2,image_size=224"
```
입력 이미지 사이즈를 변경하고 싶으면 image_size=[사이즈]를 입력하면 된다.
기본 이미지 사이즈를 사용하려면 image_size를 빼면된다.
classs의 갯수가 변경되면 변경된 갯수를 num_classes에 입력하면 된다.

### f. inference and crop bboxes by tflite model
```bash
python -u infer_and_crop_tflite.py --input_image=[테스트 이미지 경로. ex) test_image/* ] \
--output_image_dir=[crop된 이미지 저장 디렉토] --min_score_thresh=[crop할 최소 score threshold, ex: 0.6] \
--tflite_path=[tflite 모델 파일 경로]
```

## 예측한 이미지로 데이터셋 만들기

### 0. 테스트할 이미지 수집하여 한 디렉토리에 저장하기

### a. tflite 모델로 inference하기
```bash
python -u infer_and_crop_tflite.py --input_image=[테스트 이미지 경로. ex) test_image/* ] \
--output_image_dir=[crop된 이미지 저장할 디렉토리] --min_score_thresh=[crop할 최소 score threshold, ex: 0.6] \
--tflite_path=[tflite 모델 파일 경로, ex) model.tflite]
```

### b. 예측한 이미지 검수하기(정상 예측과 오탐 분류)
예측후 crop된 결과 이미지들을 하나씩 보면서 정상 예측과 오탐을 서로 다른 디렉토리로 분류
아래 예제처럼 동일한 디렉토리 아래 각각의 디렉토리로 저장하면 됨.
ex) infer/0_true_positive, infer/1_false_positive
정상적으로 예측된 이미지도 추가적으로 학습할 경우 정상 예측 이미지 디렉토리 앞에 0을 붙혀야함
(나중에 라벨 번호를 저장할때 디렉토리명으로 소팅해서 순서대로 라벨번호를 부여하기 때문에 정상 이미지는 1번 라벨로 지정하기 위함)

### c. 예측한 이미지에서 좌표정보 뽑아내어 json형태로 저장하기
```bash
python infer_and_extract_bbox_tflite.py --input_image=[crop된 이미지 경로, ex)crop_images/*/*] \
--output_path=[출력 json파일 경로, ex)eye_annotation.json] \
--real_image_dir=[위에서 예측한 원본 테스트 이미지 디렉토리] --min_score_thresh=[위에서 예측할때 사용한 값과 동일, ex)0.5] \
--tflite_path=[위에서 예측한 tflite모델과 동일] --output_image_dir=[좌표정보 뽑아낸 원본 이미지 파일 카피할 디렉토리] \
--vis_image_dir=[좌표정보 시각화한 이미지 저장할 디렉토리] --target_label_idx=1\ 
--start_index=[시작 index값, 정상 예측한 이미지도 좌표정보 뽑아내려면 1, 오탐한 이미지 좌표정보만 뽑아내려면 2] \
--label_dir=[crop된 이미지 경로의 상위 디렉토리, ex)crop_images, --input_image에 입력한 경로의 상위 디렉토리]
```

### d. 좌표정보 뽑아낸 json파일을 coco 포맷으로 변형하기
```bash
python afp_to_coco.py \
--output_path=[coco 포맷의 출력 json파일 경로, ex)coco.json] \
--annotation_files=[위에서 좌표정보 뽑아낸 json파일 경로, ex)eye_annotation.json] \
--label_dir=[위 c.와 동일한 값] \
--start_idx=[위 c.와 동일한 값]
```

### e. 최종 학습가능한 tfrecord 포맷으로 변형하기
```bash
python -m dataset.create_coco_tfrecord \
--image_dir=[위 a.에서 테스트 이미지로 사용한 디렉토리명] \
--object_annotations_file=[위 d.에서 만들어낸 coco 포맷 파일경로, ex)coco.json] \
--output_file_prefix=[tfrecord 출력 경로, ex)tfrecord/train] \
--num_shards=4
```

## 1. About EfficientDet Models

EfficientDets are a family of object detection models, which achieve state-of-the-art 53.7mAP on COCO test-dev, yet being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. Our models also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors.


EfficientDets are developed based on the advanced backbone, a new BiFPN, and a new scaling technique:

<p align="center">
<img src="./g3doc/network.png" width="800" />
</p>

  * **Backbone**: we employ [EfficientNets](https://arxiv.org/abs/1905.11946) as our backbone networks.
  * **BiFPN**: we propose BiFPN, a bi-directional feature network enhanced with fast normalization, which enables easy and fast feature fusion.
  * **Scaling**: we use a single compound scaling factor to govern the depth, width, and resolution for all backbone, feature & prediction networks.

Our model family starts from EfficientDet-D0, which has comparable accuracy as [YOLOv3](https://arxiv.org/abs/1804.02767). Then we scale up this baseline model using our compound scaling method to obtain a list of detection models EfficientDet-D1 to D6, with different trade-offs between accuracy and model complexity.


<table border="0">
<tr>
    <td>
    <img src="./g3doc/flops.png" width="100%" />
    </td>
    <td>
    <img src="./g3doc/params.png", width="100%" />
    </td>
</tr>
</table>

** For simplicity, we compare the whole detectors here. For more comparison on FPN/NAS-FPN/BiFPN, please see Table 4 of our [paper](https://arxiv.org/abs/1911.09070).



## 2. Pretrained EfficientDet Checkpoints

We have provided a list of EfficientDet checkpoints and results as follows:

|       Model    | AP<sup>test</sup>    |  AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>S</sub>   |  AP<sub>M</sub>    |  AP<sub>L</sub>   |  AP<sup>val</sup> | | #params | #FLOPs |
|----------     |------ |------ |------ | -------- | ------| ------| ------ |------ |------ |  :------: |
|     EfficientDet-D0 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/val/d0_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/testdev/d0_coco_test-dev2017.txt))    | 33.8 | 52.2 | 35.8 | 12.0 | 38.3 | 51.2 | 33.5 |  | 3.9M | 2.54B  |
|     EfficientDet-D1 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d1.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/val/d1_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/testdev/d1_coco_test-dev2017.txt))    | 39.6 | 58.6 | 42.3 | 17.9 | 44.3 | 56.0 | 39.1 | | 6.6M | 6.10B |
|     EfficientDet-D2 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d2.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/val/d2_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/testdev/d2_coco_test-dev2017.txt))    | 43.0 | 62.3 | 46.2 | 22.5 | 47.0 | 58.4 | 42.5 | | 8.1M | 11.0B |
|     EfficientDet-D3 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d3.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/val/d3_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/testdev/d3_coco_test-dev2017.txt))    | 45.8 | 65.0 | 49.3 | 26.6 | 49.4 | 59.8 | 45.9 | | 12.0M | 24.9B |
|     EfficientDet-D4 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d4.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/val/d4_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/testdev/d4_coco_test-dev2017.txt))    | 49.4 | 69.0 | 53.4 | 30.3 | 53.2 | 63.2 | 49.0 |  | 20.7M | 55.2B |
|     EfficientDet-D5 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d5.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/val/d5_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/testdev/d5_coco_test-dev2017.txt))    | 50.7 | 70.2 | 54.7 | 33.2 | 53.9 | 63.2 | 50.5 |  | 33.7M | 135.4B |
|     EfficientDet-D6 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d6.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/val/d6_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/testdev/d6_coco_test-dev2017.txt))    | 51.7 | 71.2 | 56.0 | 34.1 | 55.2 | 64.1 | 51.3 | | 51.9M  |  225.6B  |
|     EfficientDet-D7 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d7.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/val/d7_coco_val_softnms.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/testdev/d7_coco_test-dev2017_softnms.txt))    | 53.7 | 72.4 | 58.4 | 35.8 | 57.0 | 66.3 | 53.4 | | 51.9M  |  324.8B  |

** <em>val</em> denotes validation results, <em>test-dev</em> denotes test-dev2017 results. AP<sup>val</sup> is for validation accuracy, all other AP results in the table are for COCO test-dev2017. All accuracy numbers are for single-model single-scale without ensemble or test-time augmentation.  EfficientDet-D0 to D6 are trained for 300 epochs and tested with hard NMS, D7 is trained for 600 epochs and tested with soft-NMS (nms_config={"method":"gaussian"}).


## 3. Export SavedModel, frozen graph, tensort models, or tflite.

Run the following command line to export models:

    !rm  -rf savedmodeldir
    !python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 \
      --ckpt_path=efficientdet-d0 --saved_model_dir=savedmodeldir \
      --tensorrt=FP32  --tflite_path=efficientdet-d0.tflite

Then you will get:

 - saved model under `savedmodeldir/`
 - frozen graph with name `savedmodeldir/efficientdet-d0_frozen.pb`
 - TensorRT saved model under `savedmodeldir/tensorrt_fp32/`
 - tflite file with name `efficientdet-d0.tflite`

Notably, --tflite_path only works after 2.3.0-dev20200521

### converting ckpt model to tflite model
```bash
python model_inspect.py \
--runmode=saved_model \
--model_name=efficientdet-d0 \
--ckpt_path=[trained_model_ckpt_dir] \
--saved_model_dir=[saved_model_dir] \
--tflite_path=[tflite_output_path] \
--hparams="num_classes=2"
```

### tflite model test in python
```python
import numpy as np
import tensorflow as tf
from PIL import Image

model_path = "mymodel.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
im = Image.open("myimage.jpg").convert("RGB")
o_w, o_h =im.size
im = np.array(im.resize((input_shape[2], input_shape[1])))
im = np.expand_dims(im, axis=0)
interpreter.set_tensor(input_details[0]['index'], im)
import time
start = time.time()
interpreter.invoke()
print(time.time() - start)
output_data = interpreter.get_tensor(output_details[0]['index'])
print(time.time() - start)
thres = 0.3
r_h = input_shape[1]
r_w = input_shape[2]
eye_indexes = np.squeeze(np.argwhere(output_data[0,:,6] == 1), 1)
eyes = []
if len(eye_indexes) > 0:
    top_k = 10
    top_k_indexes = output_data[0][eye_indexes][:,5].argsort()[::-1][:top_k]
    scores = output_data[0][top_k_indexes][:, 5]
    bboxes = output_data[0][top_k_indexes][:, 1:5]
    for i, score in enumerate(scores):
        if score < thres:
            break
        r_y1, r_x1, r_y2, r_x2 = bboxes[i]
        
        y1 = r_y1 / r_h * o_h
        y2 = r_y2 / r_h * o_h
        x1 = r_x1 / r_w * o_w
        x2 = r_x2 / r_w * o_w
        
        eyes.append([x1,y1,x2,y2])    
    print(eyes)
else:
    print("no eyes")

```

## 4. Benchmark model latency.


There are two types of latency: network latency and end-to-end latency.

(1) To measure the network latency (from the fist conv to the last class/box
prediction output), use the following command:

    !python model_inspect.py --runmode=bm --model_name=efficientdet-d0

** add --hparams="mixed_precision=True" if running on V100.

On single Tesla V100 without TensorRT, our D0 network (no pre/post-processing)
has 134 FPS (frame per second) for batch size 1, and 238 FPS for batch size 8.

(2) To measure the end-to-end latency (from the input image to the final rendered
new image, including: image preprocessing, network, postprocessing and NMS),
use the following command:

    !rm  -rf /tmp/benchmark/
    !python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 \
      --ckpt_path=efficientdet-d0 --saved_model_dir=/tmp/benchmark/ \

    !python model_inspect.py --runmode=saved_model_benchmark \
      --saved_model_dir=/tmp/benchmark/efficientdet-d0_frozen.pb \
      --model_name=efficientdet-d0  --input_image=testdata/img1.jpg  \
      --output_image_dir=/tmp/  \

On single Tesla V100 without using TensorRT, our end-to-end
latency and throughput are:


|       Model    |   mAP | batch1 latency |  batch1 throughput |  batch8 throughput |
| ------ | ------ | ------  | ------ | ------ |
| EfficientDet-D0 |  33.8 | 10.2ms | 97 fps | 209 fps |
| EfficientDet-D1 |  39.6 | 13.5ms | 74 fps | 140 fps |
| EfficientDet-D2 |  43.0 | 17.7ms | 57 fps | 97 fps  |
| EfficientDet-D3 |  45.8 | 29.0ms | 35 fps | 58 fps  |
| EfficientDet-D4 |  49.4 | 42.8ms | 23 fps | 35 fps  |
| EfficientDet-D5 |  50.7 | 72.5ms | 14 fps | 18 fps  |
| EfficientDet-D6 |  51.7 | 92.8ms | 11 fps | - fps  |
| EfficientDet-D7 |  53.7 | 122ms  | 8.2 fps | - fps  |

** FPS means frames per second (or images/second).

## 5. Inference for images.

    # Step0: download model and testing image.
    !export MODEL=efficientdet-d0
    !export CKPT_PATH=efficientdet-d0
    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/${MODEL}.tar.gz
    !wget https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png -O img.png
    !tar xf ${MODEL}.tar.gz

    # Step 1: export saved model.
    !python model_inspect.py --runmode=saved_model \
      --model_name=efficientdet-d0 --ckpt_path=efficientdet-d0 \
      --hparams="image_size=1920x1280" \
      --saved_model_dir=/tmp/saved_model

    # Step 2: do inference with saved model.
    !python model_inspect.py --runmode=saved_model_infer \
      --model_name=efficientdet-d0  \
      --saved_model_dir=/tmp/saved_model  \
      --input_image=img.png --output_image_dir=/tmp/
    # you can visualize the output /tmp/0.jpg


Alternatively, if you want to do inference using frozen graph instead of saved model, you can run

    # Step 0 and 1 is the same as before.
    # Step 2: do inference with frozen graph.
    !python model_inspect.py --runmode=saved_model_infer \
      --model_name=efficientdet-d0  \
      --saved_model_dir=/tmp/saved_model/efficientdet-d0_frozen.pb  \
      --input_image=img.png --output_image_dir=/tmp/

Lastly, if you only have one image and just want to run a quick test, you can also run the following command (it is slow because it needs to construct the graph from scratch):

    # Run inference for a single image.
    !python model_inspect.py --runmode=infer --model_name=$MODEL \
      --hparams="image_size=1920x1280"  --max_boxes_to_draw=100   --min_score_thresh=0.4 \
      --ckpt_path=$CKPT_PATH --input_image=img.png --output_image_dir=/tmp
    # you can visualize the output /tmp/0.jpg

Here is an example of EfficientDet-D0 visualization: more on [tutorial](tutorial.ipynb)

<p align="center">
<img src="./g3doc/street.jpg" width="800" />
</p>

## 6. Inference for videos.

You can run inference for a video and show the results online:

    # step 0: download the example video.
    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/data/video480p.mov -O input.mov

    # step 1: export saved model.
    !python model_inspect.py --runmode=saved_model \
      --model_name=efficientdet-d0 --ckpt_path=efficientdet-d0 \
      --saved_model_dir=/tmp/savedmodel

    # step 2: inference video using saved_model_video.
    !python model_inspect.py --runmode=saved_model_video \
      --model_name=efficientdet-d0 \
      --saved_model_dir=/tmp/savedmodel --input_video=input.mov

    # alternative step 2: inference video and save the result.
    !python model_inspect.py --runmode=saved_model_video \
      --model_name=efficientdet-d0   \
      --saved_model_dir=/tmp/savedmodel --input_video=input.mov  \
      --output_video=output.mov

## 7. Eval on COCO 2017 val or test-dev.

    // Download coco data.
    !wget http://images.cocodataset.org/zips/val2017.zip
    !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    !unzip val2017.zip
    !unzip annotations_trainval2017.zip

    // convert coco data to tfrecord.
    !mkdir tfrecord
    !PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py \
        --image_dir=val2017 \
        --caption_annotations_file=annotations/captions_val2017.json \
        --output_file_prefix=tfrecord/val \
        --num_shards=32

    // Run eval.
    !python main.py --mode=eval  \
        --model_name=${MODEL}  --model_dir=${CKPT_PATH}  \
        --validation_file_pattern=tfrecord/val*  \
        --val_json_file=annotations/instances_val2017.json

You can also run eval on test-dev set with the following command:

    !wget http://images.cocodataset.org/zips/test2017.zip
    !unzip -q test2017.zip
    !wget http://images.cocodataset.org/annotations/image_info_test2017.zip
    !unzip image_info_test2017.zip

    !mkdir tfrecord
    !PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py \
          --image_dir=test2017 \
          --image_info_file=annotations/image_info_test-dev2017.json \
          --output_file_prefix=tfrecord/testdev \
          --num_shards=32

    # Eval on test-dev: testdev_dir must be set.
    # Also, test-dev has 20288 images rather than val 5000 images.
    !python main.py --mode=eval  \
        --model_name=${MODEL}  --model_dir=${CKPT_PATH}  \
        --validation_file_pattern=tfrecord/testdev*  \
        --testdev_dir='testdev_output' --eval_samples=20288
    # Now you can submit testdev_output/detections_test-dev2017_test_results.json to
    # coco server: https://competitions.codalab.org/competitions/20794#participate

## 8. Finetune on PASCAL VOC 2012 with detector COCO ckpt.

Download data and checkpoints.

    # Download and convert pascal data.
    !wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    !tar xf VOCtrainval_11-May-2012.tar
    !mkdir tfrecord
    !PYTHONPATH=".:$PYTHONPATH"  python dataset/create_pascal_tfrecord.py  \
        --data_dir=VOCdevkit --year=VOC2012  --output_path=tfrecord/pascal

    # Download backbone checkopints.
    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz
    !tar xf efficientdet-d0.tar.gz 

Create a config file for the PASCAL VOC dataset called voc_config.yaml and put this in it.

      num_classes: 21
      var_freeze_expr: '(efficientnet|fpn_cells|resample_p6)'
      label_id_mapping: {0: background, 1: aeroplane, 2: bicycle, 3: bird, 4: boat, 5: bottle, 6: bus, 7: car, 8: cat, 9: chair, 10: cow, 11: diningtable, 12: dog, 13: horse, 14: motorbike, 15: person, 16: pottedplant, 17: sheep, 18: sofa, 19: train, 20: tvmonitor}

Finetune needs to use --ckpt rather than --backbone_ckpt.

    !python main.py --mode=train_and_eval \
        --training_file_pattern=tfrecord/pascal*.tfrecord \
        --validation_file_pattern=tfrecord/pascal*.tfrecord \
        --model_name=efficientdet-d0 \
        --model_dir=/tmp/efficientdet-d0-finetune  \
        --ckpt=efficientdet-d0  \
        --train_batch_size=64 \
        --eval_batch_size=64 --eval_samples=1024 \
        --num_examples_per_epoch=5717 --num_epochs=50  \
        --hparams=voc_config.yaml

If you want to do inference for custom data, you can run

    # Setting hparams-flag is needed sometimes.
    !python model_inspect.py --runmode=infer \
      --model_name=efficientdet-d0   --ckpt_path=efficientdet-d0 \
      --hparams=voc_config.yaml  \
      --input_image=img.png --output_image_dir=/tmp/

You should check more details of runmode which is written in caption-4.

## 9. Train on multi GPUs.

Install [horovod](https://github.com/horovod/horovod#id6).

Create a config file for the PASCAL VOC dataset called voc_config.yaml and put this in it.

      num_classes: 21
      var_freeze_expr: '(efficientnet|fpn_cells|resample_p6)'
      label_id_mapping: {0: background, 1: aeroplane, 2: bicycle, 3: bird, 4: boat, 5: bottle, 6: bus, 7: car, 8: cat, 9: chair, 10: cow, 11: diningtable, 12: dog, 13: horse, 14: motorbike, 15: person, 16: pottedplant, 17: sheep, 18: sofa, 19: train, 20: tvmonitor}

Download efficientdet coco checkpoint.

    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz
    !tar xf efficientdet-d0.tar.gz

Finetune needs to use --ckpt rather than --backbone_ckpt.

    !horovodrun -np <num_gpus> -H localhost:<num_gpus> python main.py --mode=train \
        --training_file_pattern=tfrecord/pascal*.tfrecord \
        --validation_file_pattern=tfrecord/pascal*.tfrecord \
        --model_name=efficientdet-d0 \
        --model_dir=/tmp/efficientdet-d0-finetune  \
        --ckpt=efficientdet-d0  \
        --train_batch_size=64 \
        --eval_batch_size=64 --eval_samples=1024 \
        --num_examples_per_epoch=5717 --num_epochs=50  \
        --hparams=voc_config.yaml
        --strategy=horovod

If you want to do inference for custom data, you can run

    # Setting hparams-flag is needed sometimes.
    !python model_inspect.py --runmode=infer \
      --model_name=efficientdet-d0   --ckpt_path=efficientdet-d0 \
      --hparams=voc_config.yaml  \
      --input_image=img.png --output_image_dir=/tmp/

You should check more details of runmode which is written in caption-4.

## 10. Training EfficientDets on TPUs.

To train this model on Cloud TPU, you will need:

   * A GCE VM instance with an associated Cloud TPU resource.
   * A GCS bucket to store your training checkpoints (the "model directory").
   * Install latest TensorFlow for both GCE VM and Cloud.

Then train the model:

    !export PYTHONPATH="$PYTHONPATH:/path/to/models"
    !python main.py --tpu=TPU_NAME --training_file_pattern=DATA_DIR/*.tfrecord --model_dir=MODEL_DIR --strategy=tpu

    # TPU_NAME is the name of the TPU node, the same name that appears when you run gcloud compute tpus list, or ctpu ls.
    # MODEL_DIR is a GCS location (a URL starting with gs:// where both the GCE VM and the associated Cloud TPU have write access.
    # DATA_DIR is a GCS location to which both the GCE VM and associated Cloud TPU have read access.


For more instructions about training on TPUs, please refer to the following tutorials:

  * EfficientNet tutorial: https://cloud.google.com/tpu/docs/tutorials/efficientnet
  * RetinaNet tutorial: https://cloud.google.com/tpu/docs/tutorials/retinanet

NOTE: this is not an official Google product.
