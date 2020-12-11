import tensorflow as tf
import constant
from func import get_batch,get_trigger_feeddict,f_score,GAC_func,Cudnn_RNN,matmuls
import numpy as np

from confusion_loss import f1_confusion_loss

class Trigger_Model():
    def __init__(self,t_data,maxlen,wordemb,stage="MOGANED"):
        self.t_train,self.t_dev,self.t_test = t_data
        self.maxlen = maxlen
        self.wordemb = wordemb
        self.stage = stage
        self.build_graph()
    
    def build_graph(self):
        print('--Building Trigger MOGANED Graph--')
        self.build_GAT()

    def build_GAT(self,scope='MOGANED_Trigger'):
        maxlen = self.maxlen
        num_class = len(constant.EVENT_TYPE_TO_ID)
        keepprob = constant.t_keepprob
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Initialize'):
                posi_mat = tf.concat(
                            [tf.zeros([1,constant.posi_embedding_dim],tf.float32),
                            tf.get_variable('posi_emb',[2*maxlen,constant.posi_embedding_dim],tf.float32,initializer=tf.contrib.layers.xavier_initializer())],axis=0)
                word_mat = tf.concat([
                            tf.zeros((1, constant.embedding_dim),dtype=tf.float32),
                            tf.get_variable("unk_word_embedding", [1, constant.embedding_dim], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()),
                            tf.get_variable("wordemb", initializer=self.wordemb,trainable=True)], axis=0)
                pos_mat = tf.concat([
                            tf.zeros((1, constant.pos_dim),dtype=tf.float32),
                            tf.get_variable("pos_embedding", [len(constant.POS_TO_ID)-1, constant.pos_dim], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())],axis=0)
                ner_mat = tf.concat([
                            tf.zeros((1, constant.ner_dim),dtype=tf.float32),
                            tf.get_variable("ner_embedding", [len(constant.NER_TO_ID)-1, constant.ner_dim], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())],axis=0)
            
            with tf.variable_scope("Placeholder"):
                self.sents = sents = tf.placeholder(tf.int32,[None,maxlen],'sents')
                self.posis = posis = tf.placeholder(tf.int32,[None,maxlen],'posis')
                self.maskls = maskls = tf.placeholder(tf.float32,[None,maxlen],'maskls')
                self.maskrs = maskrs = tf.placeholder(tf.float32,[None,maxlen],'maskrs')
                self._labels = _labels = tf.placeholder(tf.int32,[None],'labels')
                labels = tf.one_hot(_labels,num_class)
                self.is_train = is_train = tf.placeholder(tf.bool,[],'is_train')
                self.lexical = lexical = tf.placeholder(tf.int32,[None,3],'lexicals')

                self.ner_idx = ner_idx = tf.placeholder(tf.int32,[None,maxlen],'ner_tags')
                self.pos_idx = pos_idx = tf.placeholder(tf.int32,[None,maxlen],'pos_tags')

                self.subg_a =  tf.sparse_placeholder(tf.float32,[None,maxlen,maxlen],'subg')

                self.subg_b =  tf.sparse_transpose(self.subg_a,[0,2,1])

                #新增的变量
                self.is_negative = tf.placeholder(tf.float32,[None])

                subg_a = tf.sparse_tensor_to_dense(self.subg_a,validate_indices=False)
                subg_b = tf.sparse_tensor_to_dense(self.subg_b,validate_indices=False)

                self.gather_idxs = tf.placeholder(tf.int32,[None,2],'gather_idxs')

                sents_len = tf.reduce_sum(tf.cast(tf.cast(sents,tf.bool),tf.int32),axis=1)
                sents_mask = tf.expand_dims(tf.sequence_mask(sents_len,maxlen,tf.float32),axis=2)

                eyes = tf.tile(tf.expand_dims(tf.eye(maxlen),0),[tf.shape(pos_idx)[0],1,1])

            with tf.variable_scope("Embedding"):
                sents_emb = tf.nn.embedding_lookup(word_mat,sents)
                posis_emb  = tf.nn.embedding_lookup(posi_mat,posis)
                pos_emb = tf.nn.embedding_lookup(pos_mat,pos_idx)
                ner_emb = tf.nn.embedding_lookup(ner_mat,ner_idx)
                concat_emb = tf.concat([sents_emb,posis_emb,pos_emb,ner_emb],axis=2)

            with tf.variable_scope("Lstm_layer"):
                rnn = Cudnn_RNN(num_layers=1, num_units=constant.hidden_dim, keep_prob=keepprob, is_train=self.is_train)
                ps,_ = rnn(concat_emb, seq_len=sents_len, concat_layers=False,keep_prob=keepprob,is_train=self.is_train)
            
            with tf.variable_scope("GAC"):
                hs = []
                for layer in range(1,constant.K+1):
                    h_layer= GAC_func(ps,matmuls(subg_a,layer),maxlen,'a',layer)+GAC_func(ps,matmuls(subg_b,layer),maxlen,'b',layer)+GAC_func(ps,eyes,maxlen,'c',layer)
                    hs.append(h_layer)

            with tf.variable_scope("Aggregation"):
                s_ctxs = []
                for layer in range(1,constant.K+1):
                    s_raw = tf.layers.dense(hs[layer-1],constant.s_dim,name='Wawa')
                    s_layer = tf.nn.tanh(s_raw)
                    ctx_apply = tf.layers.dense(s_layer,1,name='ctx',use_bias=False)
                    s_ctxs.append(ctx_apply)
                vs = tf.nn.softmax(tf.concat(s_ctxs,axis=2),axis=2) #[None,maxlen,3]
                h_concats = tf.concat([tf.expand_dims(hs[layer],2) for layer in range(constant.K)],axis=2)
                final_h = tf.reduce_sum(tf.multiply(tf.expand_dims(vs,3),h_concats),axis=2)
                gather_final_h = tf.gather_nd(final_h,self.gather_idxs)
            
            with tf.variable_scope('classifier'):
                bias_weight = (constant.t_bias_lambda-1)*(1-tf.cast(tf.equal(_labels,0),tf.float32))+1
                self.logits = logits = tf.layers.dense(gather_final_h,num_class,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer(),name='Wo')
                self.pred = pred = tf.nn.softmax(logits,axis=1)
                self.pred_label = pred_label = tf.argmax(pred,axis=1)
                #self.loss = loss = tf.reduce_sum(bias_weight*tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits),axis=0)/tf.reduce_sum(bias_weight,axis=0)
                #新增的变量
                wrong_confusion_matrix = [[0.0] * num_class] *num_class
                correct_class_weight = [1.0] * num_class
                # positive_idx = self.is_negative
                # negative_idx = 1- self.is_negative
                positive_idx = 15.0
                negative_idx = 1-positive_idx 
                self.loss = loss = tf.reduce_mean(f1_confusion_loss(_labels, logits,positive_idx, negative_idx, correct_class_weight, wrong_confusion_matrix, num_class))
                self.train_op = train_op = tf.train.AdamOptimizer(constant.t_lr).minimize(loss)
                
                
    def train_trigger(self):
        train,dev,test = self.t_train,self.t_dev,self.t_test
        saver = tf.train.Saver()
        maxlen = self.maxlen
        print('--Training Trigger--')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            devbest = 0
            testbest = (0,0,0)
            from tqdm import tqdm
            for epoch in tqdm(range(constant.t_epoch)):
                loss_list =[]
                for batch in get_batch(train,constant.t_batch_size,True):
                    loss,_ = sess.run([self.loss,self.train_op],feed_dict=get_trigger_feeddict(self,batch,self.stage,maxlen))
                    loss_list.append(loss)
                print('epoch:{}'.format(str(epoch)))
                print('loss:',np.mean(loss_list))

                pred_labels = []
                for batch in get_batch(dev,constant.t_batch_size,False):
                    pred_label = sess.run(self.pred_label,feed_dict=get_trigger_feeddict(self,batch,self.stage,maxlen,is_train=False))
                    pred_labels.extend(list(pred_label))
                golds = list(dev[0][4])
                dev_p,dev_r,dev_f = f_score(pred_labels,golds)
                print("dev_Precision: {} dev_Recall:{} dev_F1:{}".format(str(dev_p),str(dev_r),str(dev_f)))

                pred_labels = []
                for batch in get_batch(test,constant.t_batch_size,False):
                    pred_label = sess.run(self.pred_label,feed_dict=get_trigger_feeddict(self,batch,self.stage,maxlen,is_train=False))
                    pred_labels.extend(list(pred_label))
                golds = list(test[0][4])
                test_p, test_r, test_f = f_score(pred_labels, golds)
                print("test_Precision: {} test_Recall:{} test_F1:{}\n".format(str(test_p), str(test_r), str(test_f)))

                if dev_f>devbest:
                    devbest = dev_f
                    testbest = (test_p, test_r, test_f)
                    saver.save(sess,"saved_models/trigger.ckpt")
            test_p, test_r, test_f = testbest
            print("test best Precision: {} test best Recall:{} test best F1:{}".format(str(test_p), str(test_r), str(test_f)))