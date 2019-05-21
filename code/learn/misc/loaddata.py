import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR = 'PetImages'
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 50



training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open('X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open('y.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in = open('X.pickle','rb')
X = pickle.load(pickle_in)

############## TF.DATA #################
# extract from filesystem
files = tf.data.Dataset.list_files(file_pattern)
dataset = tf.data.TFRecordDataset(files)

# transformations
dataset = dataset.shuffle(10000)
dataset = dataset.repeat(NUM_EPOCHS)
dataset = dataset.map(lambda x: tf.parse_single_example(x,features))
dataset = dataset.batch(BATCH_SIZE)

# load
iterator = dataset.make_one_shot_iterator()
features = iterator.get_next()

####### BETTER TF.DATA ##########
files = tf.data.Dataset.list_files(file_pattern)
dataset = tf.data.TFRecordDataset(files,num_parallel_reads=32)

dataset = dataset.apply(tf.contrib.data.shuffle_and_repead(10000, NUM_EPOCHS))
dataset = dataset.apply(tf.conrtib.data.map_and_batch(lambda x: ..., BATCH_SIZE))

dataset = dataset.apply(tf.contrib.data.prefetct_to_device('/gpu:0'))
iterator = dataset.make_one_shot_iterator()
features = iterator.get_next()

############## NOW! #####################
import numpy
import os
import sys
import cv2
import random
import matplotlib.pyplot as plt

import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img,(150,150),interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def createDataRecord(out_filename,addrs,labels):
    # open the TFRecords filename
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # progress feedback
        if not i % 1000:
            print('Train data: {}/{}'.format(i,len(addrs)))
            sys.stdout.flush()

        img = load_image(addrs[i])

        label = labels[i]

        if img is None:
            continue

        #Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }

        #Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerialiseToString())

    writer.close()
    sys.stdout.flush()

cat_dog_train_path = 'PetImages/*/*.jpg'
addrs = glob.glob(cat_dog_train_path)
labels = [0 if 'Cat' in addr else 1 for addr in addrs]

# shuffle data
c = list(zip(addrs,labels))
shuffle(c)
addrs,labels = zip(*c)

# divide data into train,validation and test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addr))]
val_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = lables[int(0.8*len(labels)):]

createDataRecorde('train.tfrecords', train_addrs,train_labels)
createDataRecord('val.tfrecords')
createDataRecord('test.tfrecords')

################### LOADING DATA and TRAINING ################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def parser(record):
    keys_to_features = {
        'image_raw': tf.FixedLenFeature([],tf.string),
        'label': tf.FixedLenFeature([],tf.int64)
    }
    parsed = tf.parse_single_example(record,keys_to_features)
    image = tf.decode_raw(parsed['image_raw'],tf.uint8)
    image = tf.cast(image,tf.float32)
    image = tf.reshape(image, shape=[150,150,3])
    label=tf.cast(parsed['label'],tf.int32)

    return image,label

def input_fn(filenames,train,batch_size,buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser)

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        # always same evaluation (order)
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()

    x = {'image': images_batch}
    y = labels_batch

    return x,y

def train_input_fn():
    return input_fn(filenames=['train.tfrecords','test.tfrecords'],train=True)

def val_input_fn():
    return input_fn(filenames=['val.tfrecords'],train=False)

def show_images():
    features,labels =  train_input_fn()

    sess.run(train_iterator.initalizer)
    sess.run(val_iterator.initalizer)

    img, label = sess.sun([featuers['image'],labels])
    print(img.shape,label.shape)

    for i in range(img.shape[0]):
        cv2.imshow('image',img[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Class label ' + str(np.argmx(label[i])))

feature_columns = [tf.feature_column.numeric_column('image', shape=[150,150,3])]

# some bad model (ignore)
classifier = tf.estimator.DNNClassifier(
    features_columns=feature_columns,
    hidden_units=[128,64,32,16],
    activation_fn=tf.nn.relu,
    n_classes=3,
    model_dir='model',
    optimizer=tf.train.AdamOptimizer(1e-4),
    dropout=0.1
)

classifier.train(input_fn=train_input_fn, steps=100000)
result = classifier.evaluate(input_fn=val_input_fn)

print(result)

# better model
def model_fn(features, labels, mode, params):
    num_classes = 3
    net = features["image"]

    net = tf.identity(net, name="input_tensor")
    
    net = tf.reshape(net, [-1, 224, 224, 3])    

    net = tf.identity(net, name="input_tensor_after")

    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)  

    net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)    

    net = tf.contrib.layers.flatten(net)

    net = tf.layers.dense(inputs=net, name='layer_fc1',
                        units=128, activation=tf.nn.relu)  
    
    net = tf.layers.dropout(net, rate=0.5, noise_shape=None, 
                        seed=None, training=(mode == tf.estimator.ModeKeys.TRAIN))

    net = tf.layers.dense(inputs=net, name='layer_fc_2',
                        units=num_classes)

    logits = net
    y_pred = tf.nn.softmax(logits=logits)

    y_pred = tf.identity(y_pred, name="output_pred")

    y_pred_cls = tf.argmax(y_pred, axis=1)

    y_pred_cls = tf.identity(y_pred_cls, name="output_cls")


    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        metrics = {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
        }

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec

model = tf.estimator.Estimator(model_fn=model_fn,
                               params={"learning_rate": 1e-4},
                               model_dir="./model5/")

count = 0
while (count < 100000):
    model.train(input_fn=train_input_fn, steps=1000)
    result = model.evaluate(input_fn=val_input_fn)
    print(result)
    print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
    sys.stdout.flush()
count = count + 1
