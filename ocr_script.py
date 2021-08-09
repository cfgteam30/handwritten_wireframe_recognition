import tensorflow as tf

g1=tf.compat.v1.Graph()

sess = tf.compat.v1.Session(graph=g1)

from typing import Tuple
from collections import namedtuple
import pickle
import random
import cv2
import lmdb
import numpy as np
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')

class Preprocessor:
    def __init__(self,
                 img_size: Tuple[int, int],
                 padding: int = 0,
                 dynamic_width: bool = False,
                 data_augmentation: bool = False,
                 line_mode: bool = False) -> None:
        # dynamic width only supported when no data augmentation happens
        assert not (dynamic_width and data_augmentation)
        # when padding is on, we need dynamic width enabled
        assert not (padding > 0 and not dynamic_width)

        self.img_size = img_size
        self.padding = padding
        self.dynamic_width = dynamic_width
        self.data_augmentation = data_augmentation
        self.line_mode = line_mode

    @staticmethod
    def _truncate_label(text: str, max_text_len: int) -> str:
        """
        Function ctc_loss can't compute loss if it cannot find a mapping between text label and input
        labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        If a too-long label is provided, ctc_loss returns an infinite gradient.
        """
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def _simulate_text_line(self, batch: Batch) -> Batch:
        """Create image of a text line by pasting multiple word images into an image."""

        default_word_sep = 30
        default_num_words = 5

        # go over all batch elements
        res_imgs = []
        res_gt_texts = []
        for i in range(batch.batch_size):
            # number of words to put into current line
            num_words = random.randint(1, 8) if self.data_augmentation else default_num_words

            # concat ground truth texts
            curr_gt = ' '.join([batch.gt_texts[(i + j) % batch.batch_size] for j in range(num_words)])
            res_gt_texts.append(curr_gt)

            # put selected word images into list, compute target image size
            sel_imgs = []
            word_seps = [0]
            h = 0
            w = 0
            for j in range(num_words):
                curr_sel_img = batch.imgs[(i + j) % batch.batch_size]
                curr_word_sep = random.randint(20, 50) if self.data_augmentation else default_word_sep
                h = max(h, curr_sel_img.shape[0])
                w += curr_sel_img.shape[1]
                sel_imgs.append(curr_sel_img)
                if j + 1 < num_words:
                    w += curr_word_sep
                    word_seps.append(curr_word_sep)

            # put all selected word images into target image
            target = np.ones([h, w], np.uint8) * 255
            x = 0
            for curr_sel_img, curr_word_sep in zip(sel_imgs, word_seps):
                x += curr_word_sep
                y = (h - curr_sel_img.shape[0]) // 2
                target[y:y + curr_sel_img.shape[0]:, x:x + curr_sel_img.shape[1]] = curr_sel_img
                x += curr_sel_img.shape[1]

            # put image of line into result
            res_imgs.append(target)

        return Batch(res_imgs, res_gt_texts, batch.batch_size)

    def process_img(self, img: np.ndarray) -> np.ndarray:
        """Resize to target size, apply data augmentation."""

        # there are damaged files in IAM dataset - just use black image instead
        if img is None:
            img = np.zeros(self.img_size[::-1])

        # data augmentation
        img = img.astype(np.float)
        if self.data_augmentation:
            # photometric data augmentation
            if random.random() < 0.25:
                def rand_odd():
                    return random.randint(1, 3) * 2 + 1
                img = cv2.GaussianBlur(img, (rand_odd(), rand_odd()), 0)
            if random.random() < 0.25:
                img = cv2.dilate(img, np.ones((3, 3)))
            if random.random() < 0.25:
                img = cv2.erode(img, np.ones((3, 3)))

            # geometric data augmentation
            wt, ht = self.img_size
            h, w = img.shape
            f = min(wt / w, ht / h)
            fx = f * np.random.uniform(0.75, 1.05)
            fy = f * np.random.uniform(0.75, 1.05)

            # random position around center
            txc = (wt - w * fx) / 2
            tyc = (ht - h * fy) / 2
            freedom_x = max((wt - fx * w) / 2, 0)
            freedom_y = max((ht - fy * h) / 2, 0)
            tx = txc + np.random.uniform(-freedom_x, freedom_x)
            ty = tyc + np.random.uniform(-freedom_y, freedom_y)

            # map image into target image
            M = np.float32([[fx, 0, tx], [0, fy, ty]])
            target = np.ones(self.img_size[::-1]) * 255
            img = cv2.warpAffine(img, M, dsize=self.img_size, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

            # photometric data augmentation
            if random.random() < 0.5:
                img = img * (0.25 + random.random() * 0.75)
            if random.random() < 0.25:
                img = np.clip(img + (np.random.random(img.shape) - 0.5) * random.randint(1, 25), 0, 255)
            if random.random() < 0.1:
                img = 255 - img

        # no data augmentation
        else:
            if self.dynamic_width:
                ht = self.img_size[1]
                h, w = img.shape
                f = ht / h
                wt = int(f * w + self.padding)
                wt = wt + (4 - wt) % 4
                tx = (wt - w * f) / 2
                ty = 0
            else:
                wt, ht = self.img_size
                h, w = img.shape
                f = min(wt / w, ht / h)
                tx = (wt - w * f) / 2
                ty = (ht - h * f) / 2

            # map image into target image
            M = np.float32([[f, 0, tx], [0, f, ty]])
            target = np.ones([ht, wt]) * 255
            img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)

        # transpose for TF
        img = cv2.transpose(img)

        # convert to range [-1, 1]
        img = img / 255 - 0.5
        return img

    def process_batch(self, batch: Batch) -> Batch:
        if self.line_mode:
            batch = self._simulate_text_line(batch)

        res_imgs = [self.process_img(img) for img in batch.imgs]
        max_text_len = res_imgs[0].shape[0] // 4
        res_gt_texts = [self._truncate_label(gt_text, max_text_len) for gt_text in batch.gt_texts]
        return Batch(res_imgs, res_gt_texts, batch.batch_size)


fn_char_list = './models/ocr_model/model/charList.txt'
fn_summary = './models/ocr_model/model/summary.json'

with sess.as_default(): 
  with g1.as_default():
    tf.compat.v1.disable_eager_execution()

    input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))
    char_list = list(open(fn_char_list).read())
    is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
    #CNN
    cnn_in4d = tf.expand_dims(input=input_imgs, axis=3)

    # list of parameters for the layers
    kernel_vals = [5, 5, 3, 3, 3]
    feature_vals = [1, 32, 64, 128, 128, 256]
    stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
    num_layers = len(stride_vals)

    # create layers
    pool = cnn_in4d  # input to first CNN layer
    for i in range(num_layers):
        kernel = tf.Variable(
            tf.random.truncated_normal([kernel_vals[i], kernel_vals[i], feature_vals[i], feature_vals[i + 1]],
                                        stddev=0.1))
        conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
        conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=False)
        relu = tf.nn.relu(conv_norm)
        pool = tf.nn.max_pool2d(input=relu, ksize=(1, pool_vals[i][0], pool_vals[i][1], 1),
                                strides=(1, stride_vals[i][0], stride_vals[i][1], 1), padding='VALID')

    cnn_out_4d = pool

    # RNN
    rnn_in3d = tf.squeeze(cnn_out_4d, axis=[2])

    # basic cells which is used to build RNN
    num_hidden = 256
    cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in
              range(2)]  # 2 layers

    # stack basic cells
    stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    # bidirectional RNN
    # BxTxF -> BxTx2H
    (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnn_in3d,
                                                            dtype=rnn_in3d.dtype)

    # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
    concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

    # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
    kernel = tf.Variable(tf.random.truncated_normal([1, 1, num_hidden * 2, len(char_list) + 1], stddev=0.1))
    rnn_out_3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'),
                                  axis=[2])

    # BxTxC -> TxBxC
    ctc_in_3d_tbc = tf.transpose(a=rnn_out_3d, perm=[1, 0, 2])
    # ground truth text as sparse tensor
    gt_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                    tf.compat.v1.placeholder(tf.int32, [None]),
                                    tf.compat.v1.placeholder(tf.int64, [2]))

    # calc loss for batch
    seq_len = tf.compat.v1.placeholder(tf.int32, [None])
    loss = tf.reduce_mean(
        input_tensor=tf.compat.v1.nn.ctc_loss(labels=gt_texts, inputs=ctc_in_3d_tbc,
                                              sequence_length=seq_len,
                                              ctc_merge_repeated=True))

    # calc loss for each element to compute label probability
    saved_ctc_input = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, None, len(char_list) + 1])
    loss_per_element = tf.compat.v1.nn.ctc_loss(labels=gt_texts, inputs=saved_ctc_input,
                                                      sequence_length=seq_len, ctc_merge_repeated=True)

    # best path decoding or beam search decoding
    decoder = tf.nn.ctc_greedy_decoder(inputs=ctc_in_3d_tbc, sequence_length=seq_len)

    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    model_dir = './models/ocr_model/model/'
    latest_snapshot = tf.train.latest_checkpoint(model_dir)  # is there a saved model?
    saver.restore(sess, latest_snapshot)

img = cv2.imread("input.jpg",0)
with open("./label_boxes.txt") as f:
    content = f.readlines()
boxes = [list(map(int,x.strip().split(' '))) for x in content] 
imgs = [ img[y:y+h,x:x+w] for x,y,w,h in boxes ]

with sess.as_default(): 
  with g1.as_default():
    label_lines=[]
    preprocessor = Preprocessor((256,32), dynamic_width=True, padding=16)
    for img in imgs:
      img = cv2.GaussianBlur(img,(3,3),0)
      _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
      #       cv2.THRESH_BINARY,11,2)
      img = preprocessor.process_img(img)
      # imgs = list(map(preprocessor.process_img,imgs))

      # sequence length depends on input image size (model downsizes width by 4)
      max_text_len = img.shape[0] // 4
      # max_text_len = 4

      # dict containing all tensor fed into the model
      feed_dict = {
          input_imgs: [img] , 
          seq_len: [max_text_len] * 1,
          is_train: False
      }
      eval_res = sess.run([decoder,ctc_in_3d_tbc], feed_dict)
      decoded = eval_res[0][0][0]
      batch_size=1
      # contains string of labels for each batch element
      label_strs = [[] for _ in range(batch_size)]

      # go over all indices and save mapping: batch -> values
      for (idx, idx2d) in enumerate(decoded.indices):
          label = decoded.values[idx]
          batch_element = idx2d[0]  # index according to [b,t]
          label_strs[batch_element].append(label)

      # map labels to chars for all batch elements
      labels= [''.join([char_list[c] for c in labelStr]) for labelStr in label_strs]
      label_lines.append(labels[0])
with open("./labels.txt","w+") as f:
  f.write('\n'.join(label_lines))