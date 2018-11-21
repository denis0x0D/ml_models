import tensorflow as tf
import numpy as np

def lstm_gate (o, W, U, b, x, h):
  return o(tf.matmul(x, W) + tf.matmul(h, U) + bf)

def lstm_cell (Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc, xt, ht_prev, ct_prev):
   ft = lstm_gate (tf.sigmoid, Wf, Uf, bf, xt, ht_prev)
   it = lstm_gate (tf.sigmoid, Wi, Ui, bi, xt, ht_prev)
   ot = lstm_gate (tf.sigmoid, Wo, Uo, bo, xt, ht_prev)
   ct = tf.multiply (ft, ct_prev) + tf.multiply (it, lstm_gate (tf.tan, Wc, Uc, bc, xt, ht_prev))
   ht = tf.multiply (ot, tf.tan (ct_prev))
   return [ht, ct]

def init_inputs (x1, x2):
  Wf_input = np.random.rand (x1, x2)
  Uf_input = np.random.rand (x1, x2)
  bf_input = np.random.rand (1, x1)

  Wi_input = np.random.rand (x1, x2)
  Ui_input = np.random.rand (x1, x2)
  bi_input = np.random.rand (1, x1)

  Wo_input = np.random.rand (x1, x2)
  Uo_input = np.random.rand (x1, x2)
  bo_input = np.random.rand (1, x1)

  Wc_input = np.random.rand (x1, x2)
  Uc_input = np.random.rand (x1, x2)
  bc_input = np.random.rand (1, x1)

  x_input = np.random.rand (1, x1)
  h_input = np.random.rand (1, x1)
  ct_input = np.random.rand (1, x2)

  return [Wf_input, Uf_input, bf_input,
          Wi_input, Ui_input, bi_input,
          Wo_input, Uo_input, bo_input,
          Wc_input, Uc_input, bc_input,
          x_input, h_input, ct_input]

if __name__ == "__main__":
  x1 = 10
  x2 = 10
  Wf = tf.placeholder (tf.float32, shape=(x1, x2))
  Uf = tf.placeholder (tf.float32, shape=(x1, x2))
  bf = tf.placeholder (tf.float32, shape=(1, x1))
  
  Wi = tf.placeholder (tf.float32, shape=(x1, x2))
  Ui = tf.placeholder (tf.float32, shape=(x1, x2))
  bi = tf.placeholder (tf.float32, shape=(1, x1))

  Wo = tf.placeholder (tf.float32, shape=(x1, x2))
  Uo = tf.placeholder (tf.float32, shape=(x1, x2))
  bo = tf.placeholder (tf.float32, shape=(1, x1))

  Wc = tf.placeholder (tf.float32, shape=(x1, x2))
  Uc = tf.placeholder (tf.float32, shape=(x1, x2))
  bc = tf.placeholder (tf.float32, shape=(1, x1))

  xt = tf.placeholder (tf.float32, shape=(1, x1))
  ht = tf.placeholder (tf.float32, shape=(1, x1))
  ct = tf.placeholder (tf.float32, shape=(1, x2))

  [ht_next, ct_next] = lstm_cell (Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc, xt, ht, ct)
  
  with tf.Session() as sess:
    [Wf_input, Uf_input, bf_input,
    Wi_input, Ui_input, bi_input,
    Wo_input, Uo_input, bo_input,
    Wc_input, Uc_input, bc_input,
    xt_input, ht_input, ct_input] = init_inputs (x1, x2)
    print (sess.run ([ht_next, ct_next], 
      { Wf: Wf_input, Uf: Uf_input, bf: bf_input, 
      Wi: Wi_input, Ui: Ui_input, bi: bi_input,
      Wo: Wo_input, Uo: Uo_input, bo: bo_input,
      Wc: Wc_input, Uc: Uc_input, bc: bc_input,
      xt: xt_input, ht: ht_input, ct: ct_input }))
