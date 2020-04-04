import tensorflow as tf
import utils
from models import Trigger_Model
import os

flags = tf.flags
flags.DEFINE_string("gpu", "1", "The GPU to run on")
flags.DEFINE_string("mode", "GAT", "DMCNN or GAT")

def main(_):
    config = flags.FLAGS
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    extractor = utils.Extractor()
    extractor.Extract()
    loader = utils.Loader()
    t_data = loader.load_trigger()
    trigger = Trigger_Model(t_data,loader.maxlen,loader.wordemb,config.mode)
    trigger.train_trigger()

    # a_data = loader.load_argument()
    # trigger = DMCNN(t_data,a_data,loader.maxlen,loader.max_argument_len,loader.wordemb)
    # a_data_process = trigger.train_trigger()
    # argument = DMCNN(t_data,a_data_process,loader.maxlen,loader.max_argument_len,loader.wordemb,stage=config.mode)
    # argument.train_argument()


if __name__=="__main__":
    tf.app.run()