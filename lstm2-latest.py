import numpy as np
import tvm
import topi
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist/", one_hot=True)

lr = 0.001
num_steps = 1
batch_size = 64
display_step = 100

num_timesteps = 28 * 6
num_input = 28 
num_hidden = 128
num_classes = 10

sizes = [
    (num_input + num_hidden, num_hidden),
    (num_hidden,),
    (num_input + num_hidden, num_hidden),
    (num_hidden,),
    (num_input + num_hidden, num_hidden),
    (num_hidden,),
    (num_input + num_hidden, num_hidden),
    (num_hidden,),
    (num_hidden, num_classes),
    (num_classes,)
]

inits = [
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.ones, 'shape'),
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.random.normal, 'size'),
    (np.random.normal, 'size')
]

x = tvm.placeholder((batch_size, num_timesteps * num_input), 'float32')
y = tvm.placeholder((batch_size, num_classes), 'float32')
s = tvm.placeholder((batch_size, num_hidden), 'float32')
h = tvm.placeholder((batch_size, num_hidden), 'float32')

weights = [tvm.placeholder(x, 'float32', name="weights") for x in sizes]

xs = topi.split(topi.reshape(x, (batch_size, num_timesteps, num_input)), num_timesteps, axis=1)
xs = [topi.reshape(x, (batch_size, num_input)) for x in xs]
new_s = s
new_h = h

for i in range(num_timesteps):
    inp = topi.concatenate([xs[i], new_h], 1)
    g = topi.tanh(topi.matmul(inp, weights[0]) + weights[1])
    j = topi.sigmoid(topi.matmul(inp, weights[2]) + weights[3])
    f = topi.sigmoid(topi.matmul(inp, weights[4]) + weights[5])
    o = topi.sigmoid(topi.matmul(inp, weights[6]) + weights[7])

    new_s = new_s * f + g * j
    new_h = topi.tanh(new_s) * o

logits = topi.matmul(new_h, weights[8]) + weights[9]

pred = topi.nn.softmax(logits)
correct_pred = topi.equal(topi.argmax(y, 1), topi.argmax(pred, 1))
accuracy = topi.sum(correct_pred.astype('float32')) / batch_size
loss = topi.sum(-topi.sum(y * topi.nn.log_softmax(logits), axis=1)) / batch_size
head = topi.full((1,), 'float32', 1.0)

sched = tvm.create_schedule([loss.op, accuracy.op])
lowered = tvm.lower(sched, [x, y, s, h, loss, accuracy, *weights], simple_mode=True)
print (lowered)
train_model = tvm.build(sched, [x, y, s, h, loss, accuracy, *weights])
