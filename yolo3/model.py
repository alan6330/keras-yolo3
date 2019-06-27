# -*- coding: utf-8 -*
import os
import tensorflow as tf
from flyai.model.base import Base

import vgg
from path import MODEL_PATH
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model


TENSORFLOW_MODEL_DIR = "best"


class Model(Base):
    def __init__(self, data):
        self.data = data

    def create_model(self, class_num, dropout_keep_prob):
        self.vgg = vgg.VggNetModel(num_classes=class_num, dropout_keep_prob=dropout_keep_prob)

    def save_model(self, sess_info, path, name=TENSORFLOW_MODEL_DIR, overwrite=False):
        self.check(path, overwrite)
        sess = sess_info[0]
        saver = sess_info[1]
        step = sess_info[2]
        saver.save(sess, os.path.join(path, name), global_step=step)

    # 在 predict 和 predict_all 中，需要返回img_size
    def predict(self, **data):
        img_size = [224, 224]
        class_num = 24
        x = tf.placeholder(tf.float32, [1, img_size[0], img_size[1], 3])
        self.create_model(class_num, 1.)
        get_prob_bb = self.vgg.inference(x)
        # 坐标映射
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        with tf.Session() as session:
            # 加载参数
            saver.restore(session, ckpt.model_checkpoint_path)
            x_data = self.data.predict_data(**data)
            logits = session.run(get_prob_bb, feed_dict={x: x_data})
            logits[:, :12] = logits[:, :12] * img_size[1]
            logits[:, 12:] = logits[:, 12:] * img_size[0]
            return [logits, img_size]

    def predict_all(self, datas):
        model_path='trained_weights_final.h5'
        anchors_path='model_data/tiny_yolo_anchors.txt'
        classes_path='model_data/hand_classes.txt'
        score=0.3
        iou=0.45
        model_image_size=(416, 416)
        outputs = []
        for data in datas:
            box=[0 for i in range(24)]
            x_data = self.data.predict_data(**data)
            model=YOLO(model_path,anchors_path,classes_path,score,iou,model_image_size)
            out_boxes, out_scores, out_classes=model.get_boxes_scores_classes()
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                label = '{} {:.2f}'.format(predicted_class, score)
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(x_data.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(x_data.size[0], np.floor(right + 0.5).astype('int32'))
                if predicted_class=='hand_z':
                    box[0]=top
                    box[1]=bottom
                    box[12]=left
                    box[13]=right
                elif predicted_class=='hand-center':
                    box[2] = top
                    box[3] = bottom
                    box[14] = left
                    box[15] = right
                elif predicted_class=='index-finger':
                    box[4] = top
                    box[5] = bottom
                    box[16] = left
                    box[17] = right
                elif predicted_class=='middle-finger':
                    box[6] = top
                    box[7] = bottom
                    box[18] = left
                    box[19] = right
                elif predicted_class=='ring-finger':
                    box[8] = top
                    box[9] = bottom
                    box[20] = left
                    box[21] = right
                else:
                    box[10] = top
                    box[11] = bottom
                    box[22] = left
                    box[23] = right
                outputs.append(box)
            return [outputs, model_image_size]

        # img_size = [224, 224]
        # class_num = 24
        # x = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
        # self.create_model(class_num, 1.)
        # get_prob_bb = self.vgg.inference(x)
        # saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        # with tf.Session() as session:
        #     saver.restore(session, ckpt.model_checkpoint_path)
        #     outputs = []
        #     for data in datas:
        #         x_data = self.data.predict_data(**data)
        #         predict = session.run(get_prob_bb, feed_dict={x: x_data})
        #         predict[:, :12] = predict[:, :12] * img_size[1]  #
        #         predict[:, 12:] = predict[:, 12:] * img_size[0]
        #         outputs.append(predict)
        #     return [outputs, img_size]


class YOLO(object):
    def __init__(self,model_path,anchors_path,classes_path,score,iou,model_image_size,gpu_num=1):
        self.class_names = self.get_class(classes_path)
        self.anchors = self.get_anchors(anchors_path)
        self.model_path = model_path
        self.score = score
        self.iou = iou
        self.model_image_size = model_image_size
        self.gpu_num = gpu_num
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def get_class(self,classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self,anchors_path):
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def get_boxes_scores_classes(self,image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return out_boxes, out_scores, out_classes


    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()