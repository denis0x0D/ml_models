import tensorflow as tf
import time
from tensorflow.contrib.compiler import xla

IMAGE_SIZE = 28 * 28
NUM_CLASSES = 10
TRAIN_BATCH_SIZE = 100
TRAIN_STEPS = 1000

train, test = tf.keras.datasets.mnist.load_data()
train_ds = tf.data.Dataset.from_tensor_slices(train).batch(TRAIN_BATCH_SIZE).repeat()
test_ds = tf.data.Dataset.from_tensor_slices(test).batch(TRAIN_BATCH_SIZE)
iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
images, labels = iterator.get_next()
images = tf.reshape(images, [-1, IMAGE_SIZE])
images, labels = tf.cast(images, tf.float32), tf.cast(labels, tf.int64)

def build_mnist_model(x, y_):
  y = tf.keras.layers.Dense(NUM_CLASSES).apply(x)

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  return y, train_step

[y] = xla.compile(build_mnist_model, inputs=[images, labels])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
sess.run(iterator.make_initializer(train_ds))

t0 = time.time()
for i in range(TRAIN_STEPS):
  sess.run(y)

t1 = time.time()
print ("time %s" % (t1 - t0))

#sess.run(iterator.make_initializer(test_ds))
#correct_prediction = tf.equal (tf.argmax(y, 1), labels)
#accuracy = tf.reduce_mean (tf.cast(correct_prediction, tf.float32))
#print("Prediction accuracy %s" % sess.run(accuracy))
#sess.close()
 
