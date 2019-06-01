import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def findPaths(query):
    # example query: 'PetImages/*/*.jpg'
    return glob.glob(query)

def splitDataInSets():
    train_addrs = addrs[0:int(0.6*len(addrs))]
    train_labels = labels[0:int(0.6*len(labels))]
    val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addr))]
    val_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]
    test_addrs = addrs[int(0.8*len(addrs)):]
    test_labels = lables[int(0.8*len(labels)):]
    createDataRecorde('train.tfrecords', train_addrs,train_labels)
    createDataRecord('val.tfrecords')
    createDataRecord('test.tfrecords')

# THINGS TO STUDY
def study(some_list):
    np.array(some_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def betterTFData():
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = tf.data.TFRecordDataset(files,num_parallel_reads=32)

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repead(10000, NUM_EPOCHS))
    dataset = dataset.apply(tf.conrtib.data.map_and_batch(lambda x: ..., BATCH_SIZE))

    dataset = dataset.apply(tf.contrib.data.prefetct_to_device('/gpu:0'))
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

# CODE

def loadImage(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (512,512),interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # switch BGR encoding to RGB
    return np.asarray(img)

def loadImages(image_paths):
    imgs = [loadImage(path) for path in image_paths]
    return np.asarray(imgs)

def loadGreyscaleImage(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (512,512),interpolation=cv2.INTER_CUBIC)
    return img

def loadImageDirectory(path):
    imgs = []
    for filename in os.listdir(path):
        img = loadImage(os.path.join(path,filename))
        if img is not None:
            imgs.append(img)
    return np.asarray(imgs)

def loadImageDirectory_2(path):
    imgs = []
    for filepath in glob.glob(path + '*.jpg'):
        img = loadImage(filepath)
        imgs.append(img)
    return imgs

def formatImg(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for c in range(3):
                img[x,y,c] /= 255. 

def writeImage(img, path):
    cv2.imwrite(path,img)

def showCV2Image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plotImages(imgs):
    n=10
    plt.figure(figsize=(20,4))
    for i in range(len(imgs)):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def transformImage(img):
    return img.reshape(16,16)

### TF.DATA

def print_progress(count, total):
    pct = float(count) / total
    msg = '\r- Progress: {0:.1%}'.format(pct)
    sys.stdout.write(msg)
    sys.stdout.flush()

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(image_paths, out_path):
    print("Converting: " + out_path)
    num_images = len(image_paths)

    with tf.python_io.TFRecordWriter(out_path) as writer: # automatic writer.close()
        for i, path in enumerate(image_paths):
            print_progress(count=i, total=num_images-1)

            img = loadImage(path)

            # convert image to raw bytes
            img_bytes = img.tostring()

            data = {
                'image': wrap_bytes(img_bytes)
            }

            # wrap data as tf Featuers
            feature = tf.train.Features(feature=data)

            # wrap data as tf example
            example = tf.train.Example(features=feature)

            # serialize data
            serialized = example.SerializeToString()

            writer.write(serialized)

def parse(serialized):
    features = {
        'image': tf.FixedLenFeature([], tf.string)
    }

    # parse data -> dict
    parsed_example = tf.parse_single_example(serialized=serialized,features=features)

    # get image as raw bytes
    image_raw = parsed_example['image']

    # raw -> tensor
    image = tf.decode_raw(image_raw, tf.uint8)

    # uint8 -> float

    image = tf.cast(image, tf.float32)

    # tf.reshape(image, shape=[150,150,3]) ???

    return image


def input_fn(filenames, train, batch_size=32, buffer_size=2048):
    # create dataset for reading and shuffling data

    # tf.data.Dataset.list_files(file_pattern) ??
    dataset = tf.data.TFRecordDataset(filenames=filenames)


    # parse serialized data
    dataset = dataset.map(parse)

    if train:
        # read buffer of given size and randomly shuffle it
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # allow infinite reading of data
        num_repeat = None

    else:
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)

    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    images_batch = iterator.get_next()

    x = {'image': images_batch}
    y = {'image': images_batch}

    return x,y

# helper function because tf expects no arguments

def train_input_fn():
    # path_tfrecords_train = ['train.tfrecords']
    return input_fn(filenames=path_tfrecords_train, train=True)

def test_input_fn():
    # path_tfrecords_test = [test.tfrecords]
    return input_fn(filenames=path_tfrecords_test, train=False)



convert(image_paths=train_paths, out_path='train.tfrecords')


images = loadImageDirectory('../../data/testimgs/')
createTFRecord('train.tfrecords', images)
createTFRecord('test.tfrecords', images)
plotImages(images)

def loadImageIntoEstimator():
    some_images = load_images(image_paths=image_paths_test[0:9])
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'image': some_images.astype(np.float32)},
        num_epochs=1,
        shuffle=False)

def model_fn1(features, mode, params):
    x = features['image']

    net = tf.reshape(x, [-1, img_size, img_size, num_channels])

    net = tf.layers.conv2d(inputs=net, name='layer_conv1', filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # more....

    # flatten to 2-rank tensor
    net = tf.contib.layers.flatten(net)

    net = tf.layers.dense(inputs=net, name='layer_fc1', units=128, activation=tf.nn.relu)

    net = tf.layers.dense(inputs=net, name='layer_fc2', units=num_classes)

    logits = net

    y_pred = tf.nn.softmax(logits=logits)

    y_pred_cls = tf.argmax(y_pred, axis=1)

def model_fn2(features, labels, mode, params):
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

def createModelInstance():
    model = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir='./')

    model.train(input_fn=train_input_fn, steps=200)


# testimg = loadImage('../../data/sunset.jpg')
# showImage(testimg)
# print(testimg)
# arr = np.array(testimg)
# print(arr)
# formatImg(testimg)


# def program():
#     loadData()
#     transformData()
#     createTFData()
#     loadTFData()
#     createModel()
#     trainModel()
#     testModel()


