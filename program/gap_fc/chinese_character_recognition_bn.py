import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.python.ops import control_flow_ops

import matplotlib.pyplot as plt

#只用CPU计算
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# train_num = 755807
#test 223907

train_num = 19306
# 5902


logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 200, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 96, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 32020, 'the max training steps ')
# tf.app.flags.DEFINE_integer('max_steps', 20020, 'the max training steps ')
# tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
# tf.app.flags.DEFINE_integer('save_steps', 500, "the steps to save")
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 500, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('epoch', 10, 'Number of epoches')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'validation', 'Running mode. One of {"train", "valid", "test"}')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6666)
FLAGS = tf.app.flags.FLAGS



class DataIterator:
	def __init__(self, data_dir):
		# Set FLAGS.charset_size to a small value if available computation power is limited.
		truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
		print(truncate_path)
		self.image_names = []
		for root, sub_folder, file_list in os.walk(data_dir):
			if root < truncate_path:
				self.image_names += [os.path.join(root, file_path) for file_path in file_list]
		random.shuffle(self.image_names)
		self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

	@property
	def size(self):
		return len(self.labels)

	@staticmethod
	def data_augmentation(images):
		if FLAGS.random_flip_up_down:
			images = tf.image.random_flip_up_down(images)
		if FLAGS.random_brightness:
			images = tf.image.random_brightness(images, max_delta=0.3)
		if FLAGS.random_contrast:
			images = tf.image.random_contrast(images, 0.8, 1.2)
		return images

	def input_pipeline(self, batch_size, num_epochs=None, aug=False):
		images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
		labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
		input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

		labels = input_queue[1]
		images_content = tf.read_file(input_queue[0])
		images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
		if aug:
			images = self.data_augmentation(images)
		new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
		images = tf.image.resize_images(images, new_size)
		image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
														  min_after_dequeue=10000)
		# print 'image_batch', image_batch.get_shape()
		return image_batch, label_batch


def build_graph(top_k):
	keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
	images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1], name='image_batch')
	labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
	is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
	# Fast
	# with slim.arg_scope([slim.conv2d, slim.fully_connected],
	# 						activation_fn=tf.nn.relu,
	# 						normalizer_fn=slim.batch_norm,
	# 						normalizer_params={'is_training': is_training,'decay': 0.9}):
	# normalizer_params = {'is_training': is_training, 'decay': 0.95}
	with slim.arg_scope([slim.conv2d,slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
						biases_initializer=tf.constant_initializer(0.0),
						normalizer_fn=slim.batch_norm,
						normalizer_params={'is_training': is_training,'decay': 0.9}):
		# tf.summary.image('input', images, 1)
		net = slim.conv2d(images, 96, [3, 3], stride=1, padding='SAME', scope='conv1_1')
		# conv1_1_trans = slim.conv2d_transpose(net,1,[3,3])
		# tf.summary.image('conv1_1_trans', conv1_1_trans, 1)
		net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool1')
		net = slim.conv2d(net, 128, [3, 3], stride=1, padding='SAME', scope='conv2_1')
		# conv2_1_trans = slim.conv2d_transpose(net,1,[3,3])
		# tf.summary.image('conv2_1_trans', conv2_1_trans, 1)
		net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool2')
		net = slim.conv2d(net, 160, [3, 3], stride=1, padding='SAME', scope='conv3_1')
		# conv3_1_trans = slim.conv2d_transpose(net,1,[3,3])
		# tf.summary.image('conv1_1_trans', conv3_1_trans, 1)


		net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool3')

		net = slim.conv2d(net, 256, [3, 3], stride=1, padding='SAME', scope='conv4_1')
		# conv4_1_trans = slim.conv2d_transpose(net,1,[3,3])
		# tf.summary.image('conv4_1_trans', conv4_1_trans, 1)


		net = slim.conv2d(net, 256, [3, 3], stride=1, padding='SAME', scope='conv4_2')
		# conv4_2_trans = slim.conv2d_transpose(net,1,[3,3])
		# tf.summary.image('conv4_2_trans', conv4_2_trans, 1)
		net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool4')

		net = slim.conv2d(net, 384, [3, 3], stride=1, padding='SAME', scope='conv5_1')
		# conv5_1_trans = slim.conv2d_transpose(net,1,[3,3])
		# tf.summary.image('conv5_1_trans', conv5_1_trans, 1)

		net = slim.conv2d(net, 384, [3, 3], stride=1, padding='SAME', scope='conv5_2')
		# conv5_2_trans = slim.conv2d_transpose(net,1,[3,3])
		# tf.summary.image('conv5_2_trans', conv5_2_trans, 1)
		net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool5')
		#GAP
		net = slim.conv2d(net, 100, [3, 3], stride=1, padding='SAME', activation_fn=None,scope='conv6_1')
		# net = tf.reduce_mean(net, [1, 2], name='gap', keep_dims=True)
		# logits = tf.reshape(net, [-1, 100])


		nt_hpool3=tf.nn.avg_pool(net, ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1], padding='SAME')
		# net = slim.avg_pool2d(net,[3,3],stride=3,padding='SAME',scope='gap')
		
		# logits = tf.reshape(net, [-1, 100])
		
		with slim.arg_scope([slim.fully_connected],
							weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
							biases_initializer=tf.constant_initializer(0.1)):
			flatten = slim.flatten(net)
			# net = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,  scope='fc1')
			logits = slim.fully_connected(flatten, FLAGS.charset_size, activation_fn=None,scope='logits')

		# logits = tf.nn.softmax(nt_hpool3_flat)

		# net = tf.reduce_mean(net, [1, 2], name="GAP", keep_dims=True)
		# flatten = slim.flatten(net)
# with slim.arg_scope([slim.fully_connected],
# 					weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
# 					biases_initializer=tf.constant_initializer(0.1)):
# 	flatten = slim.flatten(net)
# 	net = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,  scope='fc1')
		# logits = slim.fully_connected(flatten, FLAGS.charset_size, activation_fn=None,scope='logits')

	# y=tf.nn.softmax(logits)
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
	# global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
	# optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
	# train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
	# probabilities = tf.nn.softmax(logits)
	global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
	rate = tf.train.exponential_decay(0.1, global_step, decay_steps=200, decay_rate=0.9, staircase=True)
	optimizer = tf.train.MomentumOptimizer(learning_rate=rate,momentum=0.9)
	train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
	probabilities = tf.nn.softmax(logits)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	if update_ops:
		print("BN parameters:", update_ops)
		updates = tf.group(*update_ops)
		# loss = control_flow_ops.with_dependencies([updates], loss)
		train_op = control_flow_ops.with_dependencies([updates], train_op)

	# Add summaries for BN variables
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('accuracy', accuracy)
	# for v in tf.all_variables():
	# 	if v.name.startswith('conv1_1/Batch') or v.name.startswith('conv1_2/Batch') or v.name.startswith('conv2_1/Batch') \
	# 			or v.name.startswith('conv2_2/Batch') or v.name.startswith('conv3_1/Batch') or v.name.startswith(
	# 		'conv3_2/Batch') \
	# 			or v.name.startswith('conv3_3/Batch') or v.name.startswith('conv4_1/Batch') or v.name.startswith(
	# 		'conv4_2/Batch') \
	# 			or v.name.startswith('conv4_3/Batch') or v.name.startswith('conv5_1/Batch') or v.name.startswith(
	# 		'conv5_2/Batch') \
	# 			or v.name.startswith('conv5_3/Batch') or v.name.startswith('fc6/Batch') or v.name.startswith('fc7/Batch') \
	# 			or v.name.startswith('fc8/Batch'):
	# 		print(v.name)
	# 		tf.summary.histogram(v.name, v)
	merged_summary_op = tf.summary.merge_all()

	predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
	accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

	return {'images': images,
		'labels': labels,
		'keep_prob': keep_prob,
		'top_k': top_k,
		'global_step': global_step,
		'train_op': train_op,
		'loss': loss,
		'is_training': is_training,
		'accuracy': accuracy,
		'accuracy_top_k': accuracy_in_top_k,
		'merged_summary_op': merged_summary_op,
		'predicted_distribution': probabilities,
		'predicted_index_top_k': predicted_index_top_k,
		'predicted_val_top_k': predicted_val_top_k}

def train():
	print('Begin training')
	print("一个epoch包含%d"%(train_num/FLAGS.batch_size))
	train_feeder = DataIterator(data_dir='./data/train/')
	test_feeder = DataIterator(data_dir='./data/test/')
	model_name = 'chinese-rec-model'
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
		train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
		test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
		graph = build_graph(top_k=1)
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
		# from datetime import datetime
		# TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
		# ...
		# train_log_dir = 'logs/train/' + TIMESTAMP
		# test_log_dir = 'logs/test/'   + TIMESTAMP
		# megred = tf.summary.merge_all()
		# with tf.Session() as sess:
		#     writer_train = tf.summary.FileWriter(train_log_dir,sess.graph)
		#     writer_test = tf.summary.FileWriter(test_log_dir)    
		#     ...other code...
		#     writer_train.add_summary(summary_str_train,step)
		#     writer_test.add_summary(summary_str_test,step)



		start_step = 0
		if FLAGS.restore:
			ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
			if ckpt:
				saver.restore(sess, ckpt)
				print("restore from the checkpoint {0}".format(ckpt))
				start_step += int(ckpt.split('-')[-1])

		logger.info(':::Training Start:::')
		try:
			i = 0
			while not coord.should_stop():
				i += 1
				start_time = time.time()
				train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
				feed_dict = {graph['images']: train_images_batch,
							 graph['labels']: train_labels_batch,
							 graph['keep_prob']: 0.8,
							 graph['is_training']: True}
				_, loss_val, train_summary, step = sess.run(
					[graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
					feed_dict=feed_dict)
				train_writer.add_summary(train_summary, step)
				end_time = time.time()
				logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
				if step > FLAGS.max_steps:
					break
				if step % FLAGS.eval_steps == 1:
					test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
					feed_dict = {graph['images']: test_images_batch,
								 graph['labels']: test_labels_batch,
								 graph['keep_prob']: 1.0,
								 graph['is_training']: False}
					accuracy_test, test_summary = sess.run([graph['accuracy'], graph['merged_summary_op']],
														   feed_dict=feed_dict)
					if step > 300:
						test_writer.add_summary(test_summary, step)
					logger.info('===============Eval a batch=======================')
					logger.info('the step {0} test accuracy: {1}'
								.format(step, accuracy_test))
					logger.info('===============Eval a batch=======================')
				if step % FLAGS.save_steps == 1:
					logger.info('Save the ckpt of {0}'.format(step))
					saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name),
							   global_step=graph['global_step'])
		except tf.errors.OutOfRangeError:
			logger.info('==================Train Finished================')
			saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name), global_step=graph['global_step'])
		finally:
			coord.request_stop()
		coord.join(threads)


def validation():
	print('Begin validation')
	test_feeder = DataIterator(data_dir='./data/test/')

	final_predict_val = []
	final_predict_index = []
	groundtruth = []

	with tf.Session() as sess:
		test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
		graph = build_graph(top_k=5)

		

		# tf.reset_default_graph()
		saver = tf.train.Saver()
		# tf.reset_default_graph()



		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
		if ckpt:
			saver.restore(sess, ckpt)
			print("restore from the checkpoint {0}".format(ckpt))

		logger.info(':::Start validation:::')
		try:
			i = 0
			acc_top_1, acc_top_k = 0.0, 0.0
			while not coord.should_stop():
				i += 1
				start_time = time.time()
				test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
				feed_dict = {graph['images']: test_images_batch,
							 graph['labels']: test_labels_batch,
							 graph['keep_prob']: 1.0,
							 graph['is_training']: False}
				batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'],
																	   graph['predicted_val_top_k'],
																	   graph['predicted_index_top_k'],
																	   graph['accuracy'],
																	   graph['accuracy_top_k']], feed_dict=feed_dict)
				final_predict_val += probs.tolist()
				final_predict_index += indices.tolist()
				groundtruth += batch_labels.tolist()
				acc_top_1 += acc_1
				acc_top_k += acc_k
				end_time = time.time()
				logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
							.format(i, end_time - start_time, acc_1, acc_k))

		except tf.errors.OutOfRangeError:
			logger.info('==================Validation Finished================')
			acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
			acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
			logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
		finally:
			coord.request_stop()
		coord.join(threads)
	return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}


class StrToBytes:  
	def __init__(self, fileobj):  
		self.fileobj = fileobj  
	def read(self, size):  
		return self.fileobj.read(size).encode()  
	def readline(self, size=-1):  
		return self.fileobj.readline(size).encode()

# 获取汉字label映射表
def get_label_dict():
	# f=open('./chinese_labels','r')
	# label_dict = pickle.load(f)
	# f.close()
	with open('./chinese_labels', 'r') as data_file:
		label_dict = pickle.load(StrToBytes(data_file))
		return label_dict

# 获待预测图像文件夹内的图像名字
def get_file_list(path):
	list_name=[]
	files = os.listdir(path)
	files.sort(key = lambda x : int(x[:-4]))
	print(files)
	for file in files:
		file_path = os.path.join(path, file)
		list_name.append(file_path)
	return list_name


def inference(name_list):
	print('inference')
	image_set=[]
	# 对每张图进行尺寸标准化和归一化
	for image in name_list:
		temp_image = Image.open(image).convert('L')
		temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
		temp_image = np.asarray(temp_image) / 255.0
		temp_image = temp_image.reshape([-1, FLAGS.image_size, FLAGS.image_size, 1])
		image_set.append(temp_image)
	with tf.Session() as sess:
		logger.info('========start inference============')
		# images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
		# Pass a shadow label 0. This label will not affect the computation graph.
		graph = build_graph(top_k=3)
		saver = tf.train.Saver()
		ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
		if ckpt:
			saver.restore(sess, ckpt)
		val_list=[]
		idx_list=[]
		# 预测每一张图
		for item in image_set:
			temp_image = item
			predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
											  feed_dict={graph['images']: temp_image,
														 graph['keep_prob']: 1.0,
														 graph['is_training']: False})
			val_list.append(predict_val)
			idx_list.append(predict_index)
	#return predict_val, predict_index
	return val_list,idx_list


def main(_):
	print(FLAGS.mode)
	if FLAGS.mode == "train":
		train()
	elif FLAGS.mode == 'validation':
		dct = validation()
		result_file = 'result.dict'
		logger.info('Write result into {0}'.format(result_file))
		with open(result_file, 'wb') as f:
			pickle.dump(dct, f)
		logger.info('Write file ends')
	elif FLAGS.mode == 'inference':
		label_dict = get_label_dict()
		name_list = get_file_list('./tmp')
		final_predict_val, final_predict_index = inference(name_list)
		# image_path = './data/test/00190/13320.png'
		# final_predict_val, final_predict_index = inference(image_path)
		# logger.info('the result info label {0} predict index {1} predict_val {2}'.format(190, final_predict_index,
		# 																				 final_predict_val))
		final_reco_text =[]  # 存储最后识别出来的文字串
		# 给出top 3预测，candidate1是概率最高的预测
		for i in range(len(final_predict_val)):
			candidate1 = final_predict_index[i][0][0]
			candidate2 = final_predict_index[i][0][1]
			candidate3 = final_predict_index[i][0][2]
			final_reco_text.append(label_dict[int(candidate1)])
			logger.info('[the result info] image: {0} predict: {1} {2} {3}; predict index {4} predict_val {5}'.format(name_list[i], 
				label_dict[int(candidate1)],label_dict[int(candidate2)],label_dict[int(candidate3)],final_predict_index[i],final_predict_val[i]))
		print ('=====================OCR RESULT=======================\n')
		# 打印出所有识别出来的结果（取top 1）
		for i in range(len(final_reco_text)):
		   print(final_reco_text[i],)


if __name__ == "__main__":
	tf.app.run()