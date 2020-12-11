import tensorflow as tf

class Layer():
    def __init__(self,scope,output_dim = -1,reuse = None):
        self.scope = scope
        self.reuse = reuse
        self.output_dim = output_dim
        self.call_cnt = 0
        self.initializer = None
        
        self.set_initializer()
        self.set_extra_parameters()
        self.set_extra_feeds()
        
    def __call__(self,inputs,seq_len = None):
        pass
    
    def set_extra_parameters(self,parameters = None):
        pass
    def set_extra_feeds(self,feeds = None):
        pass
    
    def check_reuse(self,scope):
        if self.call_cnt >0:
            if self.reuse == True:
                scope.reuse_variables()
            if self.reuse == None:
                if self.call_cnt >0:
                    print("Warning: Reuse variable with reuse value = None in scope",scope.name) 
                    scope.reuse_variables()
            if self.reuse == False:
                if self.call_cnt >0:
                    print("Error: reuse variable with reuse value = False in scope",scope.name) 
                    exit(-1)
        self.call_cnt +=1
    
    def set_initializer(self,initializer = None):
        if not initializer:
            initializer = tf.contrib.layers.xavier_initializer
        self.initializer = initializer

class MaskLayer(Layer):
    def __call__(self, m,seq_len):
        with tf.variable_scope(self.scope) as scope:
            self.check_reuse(scope)
            
            max_length = int(m.get_shape()[1])
            seq_len_mask = tf.sequence_mask(seq_len,maxlen = max_length, dtype = m.dtype)
            rank = m.get_shape().ndims
            extra_ones = tf.ones(rank - 2, dtype=tf.int32)
            seq_len_mask = tf.reshape(seq_len_mask, tf.concat((tf.shape(seq_len_mask), extra_ones), 0))
            if not self.mask_from_right:
                seq_len_mask = 1-seq_len_mask
            return m * seq_len_mask - ((seq_len_mask - 1) * self.mask_value)
    
    def set_extra_parameters(self,parameters = None):
        self.mask_value = 0
        self.mask_from_right = True
            
        if not parameters:
            return
        if "mask_value" in parameters:
            self.mask_value = parameters["mask_value"]
        if "mask_from_right" in parameters:
            self.mask_from_right = parameters["mask_from_right"]
    

class MaskedSoftmaxLayer(Layer):
    def __call__(self, inputs,seq_len = None):
        if inputs.dtype.is_integer:
            inputs = tf.cast(inputs,dtype = tf.float32)
        exp_val = tf.exp(inputs)
        if seq_len != None:
            with tf.variable_scope(self.scope) as scope:
                if self.call_cnt ==0:
                    self.mask = MaskLayer(scope = "Mask", reuse = self.reuse)
                self.check_reuse(scope)
                exp_val = self.mask(exp_val,seq_len)
        return exp_val / (self.epsilon + tf.reduce_sum(exp_val,axis = 1,keep_dims = True))
    
    def set_extra_parameters(self,paras =None):
        self.epsilon = 1e-8
        if paras and "epsilon" in paras:
            self.epsilon = paras["epsilon"]

def f1_confusion_loss(label_ids,logits,positive_idx,negative_idx,correct_class_weight,wrong_confusion_matrix, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    log_probs = tf.log(probs + 1e-8)
    log_one_minus_prob = tf.log(1-probs + 1e-8)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    golden_log_prob = tf.gather_nd(log_probs,label_with_idx)

    positvie_confusion_weight = tf.gather(correct_class_weight,label_ids)
    negative_confusion_weight = tf.gather(wrong_confusion_matrix,label_ids)   #B*label_size
    positive_cost = positvie_confusion_weight * golden_log_prob
    negative_cost = tf.reduce_sum(negative_confusion_weight  * log_one_minus_prob,axis = 1)
    cost = positive_cost + negative_cost

    m = tf.reduce_sum(positive_idx)
    n = tf.reduce_sum(negative_idx)
    p1 = tf.reduce_sum(positive_idx * golden_prob)
    p2 = tf.reduce_sum(negative_idx * golden_prob)
    neg_weight = p1 / (m+n-p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    balanced_weight = all_one * positive_idx + all_one * neg_weight * negative_idx
    loss = - balanced_weight * cost
    return loss



    
