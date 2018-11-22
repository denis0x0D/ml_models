import tensorflow as tf
import numpy as np
import argparse

path_to_data = "/tmp/input_data"
path_to_save_model = "/tmp/model.ckpt"
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(path_to_data, one_hot=True)

def lstm_gate (o, W, U, b, x, h):
  return o(tf.matmul(x, W) + tf.matmul(h, U) + bf)

def lstm_cell (Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc, xt, ht, ct):
   ft = lstm_gate (tf.sigmoid, Wf, Uf, bf, xt, ht)
   it = lstm_gate (tf.sigmoid, Wi, Ui, bi, xt, ht)
   ot = lstm_gate (tf.sigmoid, Wo, Uo, bo, xt, ht)
   ct = tf.multiply (ft, ct) + tf.multiply (it, lstm_gate (tf.tanh, Wc, Uc, bc, xt, ht))
   ht = tf.multiply (ot, tf.tanh (ct))
   return [ht, ct]

if __name__ == "__main__":
  num_classes = 10
  batch_size = 1
  times = 28
  num_inputs = 28
  learning_rate = 0.001
  training_steps = 1000

  Wf = tf.Variable (tf.random_normal([num_inputs, num_classes]))
  Uf = tf.Variable (tf.random_normal([num_classes, num_classes]))
  bf = tf.Variable (tf.random_normal([1, num_classes]))
  
  Wi = tf.Variable (tf.random_normal([num_inputs, num_classes]))
  Ui = tf.Variable (tf.random_normal([num_classes, num_classes]))
  bi = tf.Variable (tf.random_normal([1, num_classes]))

  Wo = tf.Variable (tf.random_normal([num_inputs, num_classes]))
  Uo = tf.Variable (tf.random_normal([num_classes, num_classes]))
  bo = tf.Variable (tf.random_normal([1, num_classes]))

  Wc = tf.Variable (tf.random_normal([num_inputs, num_classes]))
  Uc = tf.Variable (tf.random_normal([num_classes, num_classes]))
  bc = tf.Variable (tf.random_normal([1, num_classes]))
   
  ht = tf.Variable (tf.random_normal([1, num_classes]))
  ct = tf.Variable (tf.random_normal([1, num_classes]))
  
  X = tf.placeholder (tf.float32, shape=(batch_size, times, num_inputs))
  Y = tf.placeholder (tf.float32, shape=(batch_size, num_classes))

  weight = tf.Variable (tf.random_normal([num_classes, num_classes]))
  biases = tf.Variable (tf.random_normal([1, num_classes]))

  xt_times = tf.unstack (X, times, 1)
  
  for xt in xt_times:
    [ht, ct] = lstm_cell(Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc, xt, ht, ct)
  
  logits = tf.matmul(ht, weight) + biases
  # The trainig does not really work well.
  # TODO. Understand what is wrong
  prediction = tf.nn.softmax (logits)

  loss_op = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(
     logits=logits, labels=Y))

  optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  train_op = optimize.minimize(loss_op)

  correct_pred = tf.equal (tf.argmax (prediction, 1), tf.argmax (Y, 1))
  accuracy = tf.reduce_mean (tf.cast (correct_pred, tf.float32))
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  
  with tf.Session() as sess:
    sess.run (init)
    for step in range (1, training_steps):
      # get labels and images
      X_input, Y_input = mnist.train.next_batch(batch_size) 
      # reshape to batch_size * times * num_inputs
      X_input = X_input.reshape (batch_size, times, num_inputs)
      loss_op_float, accuracy_float = sess.run ([loss_op, accuracy], { X: X_input, Y: Y_input })
      if step % 200 == 0:
        print ("Step : " + str(step) + "Loss = " + "{:4f}".format(loss_op_float) + ", Accuracy = " + "{:3f}".format(accuracy_float))
    saver.save (sess, path_to_save_model)


     image_num = 11
     X_test = mnist.test.images[image_num].reshape(1, times, num_inputs)
     Y_test = mnist.test.labels[image_num].reshape(1, num_classes)
     print ("Testing Accuracy :")
     print (sess.run (accuracy , { X:X_test, Y: Y_test}))
