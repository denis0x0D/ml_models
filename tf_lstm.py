import tensorflow as tf
import numpy as np
import argparse
import time

path_to_data = "/workspace/ml_models/data"
path_to_save_model = "/examples/model_state/tf_lstm_model.ckpt"
path_to_save_graph = "/examples/model_state/"
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(path_to_data, one_hot=True)

def lstm_gate (o, W, U, b, x, h):
  return o(tf.matmul(x, W) + tf.matmul(h, U) + b)

def lstm_cell (Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc, xt, ht, ct):
  ft = lstm_gate (tf.sigmoid, Wf, Uf, bf, xt, ht)
  it = lstm_gate (tf.sigmoid, Wi, Ui, bi, xt, ht)
  ot = lstm_gate (tf.sigmoid, Wo, Uo, bo, xt, ht)
  ct = tf.multiply (ft, ct) + tf.multiply (it, lstm_gate (tf.tanh, Wc, Uc, bc, xt, ht))
  ht = tf.multiply (ot, tf.tanh (ct))
  return [ht, ct]

if __name__ == "__main__":
  num_classes = 10
  batch_size = 128
  times = 28
  num_inputs = 28
  learning_rate = 0.001
  training_steps = 1000
  Wf = tf.Variable (tf.random_normal([batch_size, num_inputs, num_classes]))
  Uf = tf.Variable (tf.random_normal([batch_size, num_classes, num_classes]))
  bf = tf.Variable (tf.random_normal([batch_size, 1, num_classes]))
  
  Wi = tf.Variable (tf.random_normal([batch_size, num_inputs, num_classes]))
  Ui = tf.Variable (tf.random_normal([batch_size, num_classes, num_classes]))
  bi = tf.Variable (tf.random_normal([batch_size, 1, num_classes]))

  Wo = tf.Variable (tf.random_normal([batch_size, num_inputs, num_classes]))
  Uo = tf.Variable (tf.random_normal([batch_size, num_classes, num_classes]))
  bo = tf.Variable (tf.random_normal([batch_size, 1, num_classes]))

  Wc = tf.Variable (tf.random_normal([batch_size, num_inputs, num_classes]))
  Uc = tf.Variable (tf.random_normal([batch_size, num_classes, num_classes]))
  bc = tf.Variable (tf.random_normal([batch_size, 1, num_classes]))
   
  ht = tf.Variable (tf.zeros([batch_size, 1, num_classes]), trainable=False)
  ct = tf.Variable (tf.zeros([batch_size, 1, num_classes]), trainable=False)
  
  X = tf.placeholder (tf.float32, shape=(batch_size, times, num_inputs))
  Y = tf.placeholder (tf.float32, shape=(batch_size, num_classes))

  weight = tf.Variable (tf.random_normal([batch_size, num_classes, num_classes]))
  biases = tf.Variable (tf.random_normal([batch_size, 1, num_classes]))
  
  jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
  with jit_scope(compile_ops=True): 
    xt_times = tf.unstack (X, times, 1)
    for xt in xt_times:
      xt = tf.reshape(xt, [batch_size, 1, num_inputs])
      [ht, ct] = lstm_cell(Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc, xt, ht, ct)
      
    logits = tf.reshape (tf.add (tf.matmul(ht, weight), biases), [batch_size, num_classes])
    # The trainig does not really work well.
    # TODO. Understand what is wrong
    prediction = tf.nn.softmax (logits, name="out_lstm_node")
    loss_op = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2(
       logits=logits, labels=Y))

    optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimize.minimize(loss_op)

    correct_pred = tf.equal (tf.argmax (prediction, 1), tf.argmax (Y, 1))
    accuracy = tf.reduce_mean (tf.cast (correct_pred, tf.float32))

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run (init)
    total_time = 0
    for step in range (1, training_steps):
      # get labels and images
      X_input, Y_input = mnist.train.next_batch(batch_size) 
    # reshape to batch_size * times * num_inputs
      X_input = X_input.reshape (batch_size, times, num_inputs)
      start = time.time()
      loss_op_float, accuracy_float, pred = sess.run ([loss_op, accuracy, correct_pred], { X: X_input, Y: Y_input })
      total_time = total_time + (time.time() - start)
    print ("TIME:")
    print (total_time)

#test_data = mnist.test.images[:batch_size].reshape((batch_size, times, num_inputs))
#    test_label = mnist.test.labels[:batch_size]#.reshape((batch_size, 1, num_classes))
#   print ("Testin accuracy ")
#    print (sess.run(accuracy, {X:test_data, Y:test_label}))
