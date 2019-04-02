import tensorflow as tf
import ssl;
import tf2xla_pb2
ssl._create_default_https_context = ssl._create_unverified_context;

tf.keras.backend.set_learning_phase(False)
model = tf.keras.applications.MobileNetV2()
model.summary()

def gen_proto_config():
  batch_size = 1
  config = tf2xla_pb2.Config()
  for x in model.inputs:
    x.set_shape([batch_size] + list(x.shape)[1:])
    feed = config.feed.add()
    feed.id.node_name = x.op.name
    feed.shape.MergeFrom(x.shape.as_proto())
    print(feed)

  for x in model.outputs:
    fetch = config.fetch.add()
    fetch.id.node_name = x.op.name

  with open('graph.config.pbtxt', 'w') as f:
    f.write(str(config))

def gen_frozen_graph():
  sess = tf.keras.backend.get_session()
  out_nodes = [node.op.name for node in model.outputs]
  graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, out_nodes)
  tf.train.write_graph(graphdef, '.', 'graph.pb', as_text=False)

if __name__ == "__main__":
  gen_proto_config()
  gen_frozen_graph()
