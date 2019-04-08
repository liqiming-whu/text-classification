import sys
import time
import argparse
import tensorflow as tf
from data_utils import *
from sklearn.model_selection import train_test_split
from attention_cnn import AttentionCNN
from base_cnn import BaseCNN

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="att_cnn3", help="base_cnn | att_cnn1 | att_cnn2 | att_cnn3")
args = parser.parse_args()

start = time.clock()

if not os.path.exists("dbpedia_csv"):
    print("Downloading dbpedia dataset...")
    download_dbpedia()

NUM_CLASS = 14
BATCH_SIZE = 64
NUM_EPOCHS = 10
WORD_MAX_LEN = 25
EMBEDDING_SIZE = 100
# GLOVE = "data/glove.6B.100d.txt"

print("Building dataset...")

word_dict = build_word_dict()
vocabulary_size = len(word_dict)
x, y = build_word_dataset("train", word_dict, WORD_MAX_LEN)
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15)


with tf.Session() as sess:
    if args.model == "base_cnn":
        model = BaseCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    elif args.model == "att_cnn1":
        model = AttentionCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS, True, False)
    elif args.model == "att_cnn2":
        model = AttentionCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS, False, True )
    elif args.model == "att_cnn3":
        model = AttentionCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS, True, True )
    else:
        raise NotImplementedError()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    '''if GLOVE:
        initW = np.random.uniform(-1.0, 1.0, (vocabulary_size, EMBEDDING_SIZE))
        glove_dic = {}
        mean = np.zeros(EMBEDDING_SIZE)
        count = 0
        with open(GLOVE, "rb") as f:
            for line in f:
                values = line.split()
                word = values[0]
                word_vec = np.array(values[1:], dtype = 'float32')
                glove_dic[word] = word_vec
                mean = mean + word_vec
                count = count + 1
            mean = mean / count

        for key in word_dict:
            word_u = key
            if word_u in glove_dic:
                initW[word_dict[word_u]] = glove[word_u]

            else:
                initW[word_dict[word_u]] = np.random.normal(mean, 0.1, size=EMBEDDING_SIZE)

        sess.run(model.W.assign(initW))
        print('GloVe loaded!')'''

    train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
    num_batches_per_epoch = (len(train_x) - 1) // BATCH_SIZE + 1
    max_accuracy = 0

    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: True
        }

        _, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict=train_feed_dict)

        if step % 100 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        if step % 500 == 0:
            run_time = time.clock()
            print("%f CPU seconds"%(run_time-start))

        if step % 2000 == 0:
            valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0

            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.x: valid_x_batch,
                    model.y: valid_y_batch,
                    model.is_training: False
                }

                accuracy = sess.run(model.accuracy, feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                cnt += 1
            valid_accuracy = sum_accuracy / cnt

            print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))

            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, "{0}/{1}.ckpt".format(args.model, args.model), global_step=step)
                print("Model is saved.\n")

        if step > 50000:
            end = time.clock()
            print("%f CPU seconds"%(end-start))
            sys.exit()

