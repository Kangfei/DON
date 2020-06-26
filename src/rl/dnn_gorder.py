#! usr/bin/python
import numpy as np
import tensorflow as tf
import os
import random
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from Queue import Queue
from threading import Thread
from scipy.stats import entropy
import time
import itertools
tf.logging.set_verbosity(tf.logging.INFO)
gpu_options = tf.GPUOptions(allow_growth=True)


###############################Models##########################################

def bulid_model(features, n_class, mode): 
  """
  The target model(DeepSet)
  Node embedding ==> Pooling ==> DNN
  """
  units = 256
  training = mode==tf.estimator.ModeKeys.TRAIN
  features = tf.cast(features, tf.float32)
  dense = features
  dense = tf.layers.dense(inputs=dense, units=units, activation=tf.nn.leaky_relu)
  logits = tf.layers.dense(inputs=dense, units = n_class, activation= None)
  return logits

def model_fn(features, labels, mode, params):
  """
  The model wrapper. 
  Input:
    features['x']: The input data.
    params['n_classes']: The number of classes.
    params['learning_rate']: The learing rate
  """
  #print(features)
  feature = features['x']
  n_class = params["n_classes"]
  lr =params["learning_rate"]
  logits = bulid_model(feature, n_class, mode)
  
  loss = None
  train_op = None
  predictions = None
  eval_metric_ops = None
  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != tf.estimator.ModeKeys.PREDICT:
    weights = tf.squeeze(features['w'])
    onehot_labels = labels
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,weights=weights, logits=logits)
    eval_metric_ops = {
        'RMSE': tf.metrics.root_mean_squared_error(labels=labels, predictions=tf.nn.softmax(logits))
    }
 
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }
  
  # Return a ModelFnOps object
  return tf.estimator.EstimatorSpec(
      mode=mode, 
      predictions=predictions, 
      eval_metric_ops=eval_metric_ops, 
      loss=loss, 
      train_op=train_op)


def data_preprocessing(locality_fname, merge_fname, load_traindata=True):
    """
    Data Preprocess               
    Return Values:                
    X_train: n_classes, nclasses-dim one-hot vectors
    Y_train: n_classes, nclasses-dim soft labels
    classMap: dictionary { classid -> set[nodeid] }
    """
    X_train = None
    Y_train = None
    classMap = None

    classMap, n_classes, n_nodes = load_classmap(merge_fname)
    
    #Do not load actural data 
    if not load_traindata:
        return (X_train, Y_train, n_classes, classMap)

    file = open(locality_fname)
    M = n_classes * 2
    N = n_classes                 
    X = np.zeros(shape = (M, N), dtype = int)
    #   Y : soft label            
    Y = np.zeros(shape = (M, N), dtype = np.float32)
    #   Cnt 
    X_norm = np.zeros(shape= (M,1), dtype = int)
    #   load the locality trainning data
    cnt = 0                       
    for line in file:             
        pairs = line.strip().split()
        node = pairs[0].split(',')                                        
                                  
        #   generate cnt-th one-hot training data   
        X[cnt][int(node[0])] = 1                         
        #   generate cnt-th soft label 
        for i in range(1,len(pairs)):
            node = pairs[i].split(',')
            idx = int(node[0])
            v = float(node[1])
            Y[cnt][idx] = v

        X_norm[cnt] = np.sum(Y[cnt])
        Y[cnt] = Y[cnt] / X_norm[cnt]
        
        if X_norm[cnt] == 0:
            print(X_norm[cnt])

        cnt = cnt + 1
                                 
    X_train = X[0:cnt]            
    Y_train = Y[0:cnt]            
    X_norm = X_norm[0:cnt]                              
    return (X_train, Y_train, X_norm, n_classes, n_nodes, classMap)

def load_classmap(merge_fname):
    """
    load the merge file to map{classid -> [nodeid]}
    """
    classMap = {}                 
    file = open(merge_fname)      
    n_classes = 0
    n_nodes = 0         
    for line in file:
        n_classes = n_classes + 1
        items = line.strip().split()
        classId = int(items[0])   
                                  
        classMap[classId] = set()
        for i in range(1, len(items)):
            classMap[classId].add(int(items[i]))
            n_nodes += 1
    return classMap, n_classes, n_nodes

def build_dataset(input_folder, load_traindata=True):
    locality_fname = os.path.join(input_folder, "locality_training.txt")
    merge_fname = os.path.join(input_folder, "merge_classes.txt")
    (X_train, Y_train, X_norm, n_classes,n_nodes, classmap) = data_preprocessing(locality_fname, merge_fname, load_traindata)
    return X_train, Y_train, n_classes, classmap, X_norm

def build_testset(input_folder):
    locality_fname = os.path.join(input_folder, "locality_training.txt")
    merge_fname = os.path.join(input_folder, "merge_classes.txt")
    (X_train, Y_train, X_norm, n_classes, n_nodes, classmap) = data_preprocessing(locality_fname, merge_fname, True)
    return X_norm, n_classes, n_nodes, classmap

def train_input_fn(X_train, Y_train, X_norm, p, set_size, shuffle=False, batch_size=2):
    """
    Construct the input_fn for Estimator.
    Params:
    X_train: data source.
    Y_train: label source.
    set_size: w-1 by default
    shuffle: The data needs to be shuffled or not.
    batch_size: Batch size of data.
    """

    def data_generator(X_train, Y_train, X_norm, p):
        """
        A generator to generate the training data
        """
        n_sample = X_train.shape[0]
        while True:
            #Importance sampling
            idx = np.random.choice(n_sample, size= set_size, replace = False, p=p)
            x = np.zeros(shape = [n_sample,])
            x[idx] = 1
            w = np.sum(X_norm[idx]).reshape((1,))
            
            s = 0
            for pair in itertools.combinations(idx, 2):
              s = s + Y_train[pair[0]][pair[1]]
            y = np.sum(np.squeeze(Y_train[idx]), axis = 0)
            y = y + s
            y = y / np.sum(y)  # re-normalize the label
            yield (x, y, w)
    
    def decode(x, y, w):
        """
        The helper function to preprocess the input data.
        Due to the input format of Estimator, the data shold be warpped by a dict. 
        """
        return {"x":x, "w":w}, y
    x_dim = X_train.shape[1]
    y_dim = Y_train.shape[1]
    
    ds = tf.data.Dataset.from_generator(lambda: data_generator(X_train, Y_train, X_norm, p), 
                                        (tf.float32, tf.float32, tf.float32),
                                        (tf.TensorShape([x_dim]), tf.TensorShape([y_dim]),tf.TensorShape([1])))
    ds = ds.map(decode)
    if shuffle:
        ds = ds.shuffle(10000)
    #ds = ds.repeat()
    ds = ds.batch(batch_size)
    iter = ds.make_one_shot_iterator()
    x, y = iter.get_next()
    return x, y

####################### Prepare the evaluation dataset #################################

def build_eval_input_fn( n_eval_data, n_classes, set_size, 
    X_train, Y_train, X_norm):
    # return: an evaluation input_fn, initial p value
    eval_data = np.zeros(shape = [n_eval_data, n_classes])
    eval_labels = np.zeros(shape = [n_eval_data, n_classes], dtype = np.float32)
    eval_norm = np.zeros(shape = [n_eval_data, 1])
    p = X_norm / float(np.sum(X_norm)) 
    p = p[:,0]
    print(p.shape)
    # choose the start vertices
    seeds = np.random.choice(n_classes, size= n_eval_data, replace = True, p=p)
    for i in range(len(seeds)):
        x, y, w = build_one_eval_input(seeds[i], n_classes, set_size, X_train, Y_train, X_norm)
        eval_data[i] = x
        eval_labels[i] = y
        eval_norm[i] = w

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_norm },
        y= eval_labels, num_epochs=1, shuffle=False)
    initial_p = initial_sampling(eval_data)
    return eval_input_fn, initial_p

# generate one eval data by greedy algorithm
def build_one_eval_input(seed, n_classes, set_size, X_train, Y_train, X_norm):
    searched = np.zeros(shape = [n_classes,])
    select_idx = [seed]
    searched[seed] = -1
    w = set_size - 1
    # generate a sequence of length set_size - 1
    total_score = 0
    for i in range(w):
        score_max = -1
        idx_max = -1
        for k in range(n_classes):
            if searched[k] < 0:
                continue
            j = i - 1
            score = 0
            while j >= 0 and j >= i - w:
                score = score + Y_train[k][select_idx[j]] 
                j = j - 1
            if score_max < score:
                score_max = score
                idx_max = k
        select_idx.append(idx_max)
        total_score += score_max
        searched[idx_max] = -1
    
    x = np.zeros(shape = [n_classes,])
    x[select_idx] = 1
    y = np.sum(np.squeeze(Y_train[select_idx]), axis = 0)
    y = y + total_score
    y = y / np.sum(y)
    w = np.sum(X_norm[select_idx]).reshape((1,))
    return x, y, w

# generate the initial p from eval data
def initial_sampling(eval_data):
    #p = np.sum(eval_data, axis = 1)
    p = np.sum(eval_data, axis = 0).astype(np.float32)
    p = p / np.sum(p)
    return p


######### Prepare the evaluation dataset (by TF pipeline) #############
def eval_input_fn(n_eval_data, n_classes, set_size, 
    X_train, Y_train, X_norm, shuffle = False, num_epochs = 1, num_parallel_calls = None):
    p = X_norm / float(np.sum(X_norm)) 
    p = p[:,0]
    # choose the start vertices
    seeds = np.random.choice(n_classes, size= n_eval_data, replace = True, p=p)

    def gorder_greedy(seed):
        searched = np.zeros(shape = [n_classes,])
        select_idx = [seed]
        searched[seed] = -1
        w = set_size - 1
        # generate a sequence of length set_size - 1
        total_score = 0
        for i in range(w):
            score_max = -1
            idx_max = -1
            for k in range(n_classes):
                if searched[k] < 0:
                    continue
                j = i - 1
                score = 0
                while j >= 0 and j >= i - w:
                    score = score + Y_train[k][select_idx[j]] 
                    j = j - 1
                if score_max < score:
                    score_max = score
                    idx_max = k
            select_idx.append(idx_max)
            total_score += score_max
            searched[idx_max] = -1
    
        x = np.zeros(shape = [n_sample,])
        x[selected_idx] = 1
        y = np.sum(np.squeeze(Y_train[selected_idx]), axis = 0)
        y = y + total_score
        y = y / np.sum(p)
        w = np.sum(X_norm[idx]).reshape((1,))
        return x, y, w

    def decode(x, y, w):
        """
        The helper function to preprocess the input data.
        Due to the input format of Estimator, the data shold be warpped by a dict. 
        """
        return {"x":x, "w":w}, y

    ds = tf.data.Dataset.from_tensor_slices(seeds)
    ds = ds.map(map_func = gorder_greedy, num_parallel_calls = num_parallel_calls).map(decode)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.repeat(num_epochs)
    iter = ds.make_one_shot_iterator()
    x, y = iter.get_next()
    return x, y


############################### Prediction ####################################

def main(args):
  fname = args.input_data_folder
  model_dir = args.model_dir
  bs = args.batch_size
  steps = args.steps
  lr = args.learning_rate
  verbose = args.verbose
  w = args.w
  
  # Data pre-processing 
  train_data, train_labels, n_classes, _, train_norm = build_dataset(fname)
  print ("training data shape: ")
  print (train_data.shape, train_labels.shape, train_norm.shape)
  
  if verbose:
      print("Real Class: %d"%(n_classes))
  # Build classifier
  classifier = tf.estimator.Estimator(model_fn=model_fn,
                                      params={"n_classes": n_classes, "learning_rate": lr},
                                      model_dir=model_dir,
                                      config=tf.estimator.RunConfig(session_config=tf.ConfigProto(gpu_options=gpu_options)))

  # Bulid Input (should pass a dict)
  input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":train_data},
                                                y=train_labels, 
                                                num_epochs=None,
                                                shuffle=True,
                                                batch_size=bs)
  # Train the model
  #classifier.train(input_fn=input_fn, steps=steps)
  classifier.train(input_fn=lambda:train_input_fn(train_data, train_labels, train_norm, set_size = w-1, shuffle=True, batch_size=bs), steps=steps)
  # Evaluation

  input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data, "w":train_norm},
                                                y=train_labels, 
                                                num_epochs=1,
                                                shuffle=False)
  eval_results = classifier.evaluate(input_fn=input_fn)
  print(eval_results)


class FastPredictor:
    def __init__(self, model_fn, model_dir, lr, n_classes):
        self.classifier = tf.estimator.Estimator(model_fn=model_fn, 
                                                 params={"n_classes":n_classes, "learning_rate": lr},
                                                 model_dir=model_dir)
        self.n_classes = n_classes
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.prediction_thread = Thread(target=self._predict_from_queue)
        self.prediction_thread.setDaemon(True)
        self.prediction_thread.start()

    def _generate_from_queue(self):
        """
        The helper function. As a data generator.
        """
        while True:
            yield self.input_queue.get()
    
    def _predict_input_fn(self):
        """
        The helper function for constructing the input_fn for prediction.
        """
        def decode(x):
            """
            The helper function to preprocess the input data.
            Due to the input format of Estimator, the data shold be warpped by a dict. 
            """
            x = tf.expand_dims(x, 0)
            return {"x":x}
        ds = tf.data.Dataset.from_generator(self._generate_from_queue, 
                                           (tf.float32),
                                           (tf.TensorShape([None, self.n_classes])))
        ds = ds.map(decode)

        iter = ds.make_one_shot_iterator()
 
        return iter.get_next()


    def _predict_from_queue(self):
        """
        The predict helper function.
        Iterate according to _predict_input_fn
        """
        for i in self.classifier.predict(input_fn=self._predict_input_fn):
            self.output_queue.put(i)

    def predict(self, feature):
        """
        The prediction function.
        feature: The feature vector.
        pred: The output predictions.
        """
        self.input_queue.put(feature)
        pred = self.output_queue.get()
        return pred

def test_FastPredictor(args):
    model_dir = args.model_dir
    lr = args.learning_rate
    fname = args.input_data_folder
    #X_norm, N, classmap =  build_dataset(fname)
    train_data, train_labels, n_classes, _ = build_dataset(fname)
    fp = FastPredictor(model_fn, model_dir, lr, n_classes)
    #res = fp.predict(np.expand_dims(train_data[1], 0))
    s = time.time()
    for t in train_data:
        res = fp.predict(t)
        #print(res)
    print("%.6f"%(time.time() - s))

def gen_order_union(args):
  # Create the Estimator
  #N = args.n_classes
  lr = args.learning_rate
  fname = args.input_data_folder
  verbose = args.verbose
  model_dir = args.model_dir
  w = args.w
  set_size = w - 1

  X_norm, n_classes, n_nodes, classmap =  build_testset(fname)
  mask = np.zeros(shape=(n_classes,),dtype=int)

  fp = FastPredictor(model_fn, model_dir, lr, n_classes)
 
  res = []
  class_res = []
  seed = np.argmax(X_norm)
  init_set = [seed]
  cnt = 0
  for i in init_set:
      real_idx = classmap[i].pop()
      if len(classmap[i])==0:
          mask[i] = -10000000.0
      res.append(real_idx)
      class_res.append(i)

  cnt = len(res)
  def one_hot(r):
      res = np.zeros(shape=(n_classes,), dtype=int)
      res[r] = 1
      return res

  while cnt < n_nodes:
      pred = np.zeros(shape=(1, n_classes), dtype=np.float32)
      # set_size * n_classes one-hot vector 
      input = np.zeros(shape=(n_classes,), dtype=int)
      input[class_res[-w+1:]] = 1
      input = input.reshape((1, n_classes))

      pred = fp.predict(input)["probabilities"]
      t_pred = pred + mask
      oidx = np.argmax(t_pred)
      #if verbose:
      #    print("cnt: %d with: %d without: %d"%(len(class_res), oidx, np.argmax(pred)))
      
      # must not raise exception, if so, something wrong with data
      real_idx = classmap[oidx].pop()

      if len(classmap[oidx]) == 0:
          mask[oidx] = -10000000.0
      res.append(real_idx)
      class_res.append(oidx)
      cnt = cnt + 1
      #if cnt % 100 ==0:
      #    break
  out = "wv_ml_order2_union.txt"
  fout = open(out, 'w')
  for r in res:
      fout.write(str(r+1)+"\n")
  fout.close()
  print "generate order union:"
  print computeF(w, datapath=args.input_data_folder, fname=out)

def gen_order_sep(args):
  # Create the Estimator
  #N = args.n_classes
  lr = args.learning_rate
  fname = args.input_data_folder
  verbose = args.verbose
  model_dir = args.model_dir
  w = args.w

  X_norm, n_classes, n_nodes, classmap =  build_testset(fname)
  mask = np.zeros(shape=(n_classes,),dtype=int)

  classifier = tf.estimator.Estimator(model_fn=model_fn, 
                                      params={"n_classes": n_classes, "learning_rate": lr},
                                      model_dir=model_dir)
 
 
  res = []
  class_res = []
  seed = np.argmax(X_norm)
  print("seed: %d\t%d"%(seed,X_norm[seed]))
  init_set = [seed]
  cnt = 0
  for i in init_set:
      real_idx = classmap[i].pop()
      if len(classmap[i])==0:
          mask[i] = -10000000.0
      res.append(real_idx)
      class_res.append(i)

  #Predict all labels
  cnt = len(res)
  data = np.identity(n_classes)
  data = tf.estimator.inputs.numpy_input_fn(x={"x": data}, num_epochs=1,shuffle=False)
  pred_all = list(classifier.predict(data)) 
  pred_all = [p["probabilities"] for p in pred_all]
  
  while cnt < n_nodes:
      pred = np.zeros(shape=(1, n_classes), dtype=int)
      coff_cnt = 0
      for r in class_res[-w:]:
          pred = pred + pred_all[r] * X_norm[r]
          coff_cnt = coff_cnt + 1
      t_pred = pred + mask
      oidx = np.argmax(t_pred)
      #if verbose:
      #    print("cnt: %d with: %d without: %d"%(len(class_res), oidx, np.argmax(pred)))
      
      # must not raise exception, if so, something wrong with data
      real_idx = classmap[oidx].pop()

      if len(classmap[oidx]) == 0:
          mask[oidx] = -10000000.0
      res.append(real_idx)
      class_res.append(oidx)
      cnt = cnt + 1
      #if cnt % 100 ==0:
      #    break
  out = "wv_ml_order2_sep.txt"
  fout = open(out, 'w')
  for r in res:
      fout.write(str(r+1)+"\n")
  fout.close()
  print "generate order seq:"
  print computeF(w, datapath=args.input_data_folder, fname=out)

def computeF(w, datapath="../../data/wv",fname="wv_ml_order2_sep.txt"):
  import subprocess
  graph_data = os.path.join(datapath, 'graph.txt')
  cmd = "../compute_F %s %s %d"%(graph_data, fname, w)
  res = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True)
  result = res.stdout.readlines()
  num = int(result[-4].strip().split(":")[1][1:])
  return "w: %d\tF: %d"%(w,num)

if __name__ == "__main__":
  parser = ArgumentParser("dnn", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
  #Required 
  parser.add_argument("--input_data_folder", default="../../data/wv", help="Input data folder")
  #parser.add_argument("--n_classes", default=7115, type=int)
  parser.add_argument("--model_dir", default="models_tmp")
  #parser.add_argument("--output_folder", default="output/wv")
  parser.add_argument("--w", default = 5, type = int)
  #Learning Parameters
  parser.add_argument("--batch_size", default=128, type=int)
  parser.add_argument("--steps", default=1000, type=int)
  parser.add_argument("--learning_rate", default = 0.001, type=float)
  #Others
  parser.add_argument("--verbose", default=False, type=bool)
  parser.add_argument("--genorder", default=True, type=bool)
 
  args = parser.parse_args() 
  print(args)
  main(args)

  gen_order_union(args)
  gen_order_sep(args)
