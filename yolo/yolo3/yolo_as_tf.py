from glob import glob

import tensorflow.compat.v1 as tf


def restore_tf_checkpoint(conf):
    tf.disable_v2_behavior()
    tf.disable_eager_execution()

    tf.reset_default_graph()

    sess = tf.InteractiveSession()
    tf_meta_path = glob('{}/*.meta'.format(conf['checkpoint_dir']))[0]
    saver = tf.train.import_meta_graph(tf_meta_path)
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, tf.train.latest_checkpoint(conf['checkpoint_dir']))

    graph = tf.get_default_graph()
    input_placeholder = graph.get_tensor_by_name(conf['input_node'])
    if isinstance(conf['output_node'], list):
        output_placeholder = [graph.get_tensor_by_name(node) for node in conf['output_node']]
    else:
        output_placeholder = graph.get_tensor_by_name(conf['output_node'])

    return {
        'sess': sess,
        'in': input_placeholder,
        'out': output_placeholder
    }


class Model:
    """
    The model which imitates keras's model behavior.
    The model can be used to do predictions and evaluations in YOLO ecosystem.
    """
    def __init__(self):
        self._tf_spec = None

    def predict_on_batch(self, x):
        """
        Runs session on input data and returns numpy array
        """
        pred = self._tf_spec['sess'].run(
            fetches=self._tf_spec['out'],
            feed_dict={
                self._tf_spec['in']: x
            }, )

        return pred

    def load(self, path):
        """
        Restore tensorflow checkpoint

        param path: path to the directory where the checkpoint is located.
        """
        conf = {
            'checkpoint_dir': path,
            'input_node': 'input_1:0',
            'output_node': ['conv_81/BiasAdd:0', 'conv_93/BiasAdd:0', 'conv_105/BiasAdd:0']
        }

        self._tf_spec = restore_tf_checkpoint(conf)


def load_model_tf(checkpoint_path):
    """
    Restores custom model class which imitates keras' Model behaviour
    """
    model = Model()

    model.load(checkpoint_path)

    return model
