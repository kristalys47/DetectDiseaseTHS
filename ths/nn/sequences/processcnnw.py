from ths.nn.sequences.tweets import TweetSentiment2DCNNv2_1, TweetSentiment2LSTM2Dense, TweetSentiment2LSTM2Dense3Layer, TweetSentiment2LSTM2Dense4Layer, TweetSentiment2LSTM2Attention, TweetSentiment2LSTM2Attentionv2
from ths.nn.sequences.cnn import TweetSentimentInceptionOneChan, KerasInceptionCNN
from ths.nn.sequences.tweets import TweetSentimentCNN

from ths.utils.files import GloveEmbedding, Word2VecEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbedding, PadSentences, TrimSentences
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.regularizers import l2
from ths.nn.metrics.f1score import f1, precision, recall, fprate
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from ths.utils.errors import ErrorAnalysis
from ths.nn.metrics.f1score import calculate_cm_metrics
import sys

import numpy as np
import csv
import math


class ProcessTweetsCNN:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def plot(self, history):
        # summarize history for accuracy
        plt.figure(1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # summarize history for loss
        plt.figure(2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def process(self, json_filename, h5_filename, plot=False, epochs = 100, vect_dimensions = 100):
        # open the file with tweets
        X_all = []
        Y_all = []
        All  = []

        #with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
        with open(self.labeled_tweets_filename, "r", encoding="utf-8") as f:
            i = 0
            csv_file = csv.reader(f, delimiter=',')
            ones_count = 0

            for r in csv_file:
                if i != 0:
                    All.append(r)
                i = i + 1

        np.random.shuffle(All)

        ones_count = 0
        two_count = 0
        zero_count = 0
        for r in All:
            tweet = r[0]
            label = int(r[1])
            if (label == 0):
                zero_count += 1
            elif (label == 1):
                ones_count += 1
            else:
                two_count += 1
            X_all.append(tweet)
            Y_all.append(label)

        print("len(Y_all): ", len(Y_all))
        class_weight_val = class_weight.compute_class_weight('balanced', np.unique(Y_all), Y_all)
        print("classes: ", np.unique(Y_all))
        print("counts for 0, 1, 2: ", zero_count, ones_count, two_count)
        print("class weight_val: ", class_weight_val)
        class_weight_dictionary = {0: class_weight_val[0], 1: class_weight_val[1], 2: class_weight_val[2]}
        print("dict: ", class_weight_dictionary)

        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.80)
        X_train_sentences = X_all
        Y_train = Y_all

        G = GloveEmbedding(self.embedding_filename, dimensions=100)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        print("hello", embedding[47])
        print("hello", embedding[9876])


        S = SentenceToEmbedding(word_to_idx, idx_to_word, embedding)

        #X_train_matrixes = S.map_sentence(c)

        edata = []
        padding_vect = [0] * 100
        #
        # s = "I love New York and music locon"
        #
        # e = "I love New York and music locon"
        #
        # mskdn = [e, s]

        for i in X_train_sentences:
            m = S.map_sentence(i)
            if len(m) < 75:
                while len(m) < 75:
                    m = np.vstack((m, padding_vect))
            edata.append(m)
        print("hello", edata[1])
        print("len", len(edata[1]), " ", len(edata[1][1])," ")

        # padding_len = self.max_len - len(sentence)
        # if (padding_len > 0):
        #     padding = []
        #     r = range(0, padding_len)
        #     for _ in r:
        #         padding.append(0)
        # return sentence + padding






        # print("Train data mappend to indices")
        # if max_len % 2 !=0:
        #     max_len = max_len + 1
        #
        # P = PadSentences(max_len)
        # X_train_pad = P.pad_list(X_train_indices)
        # print("Train data padded")
        # # TRIM
        # trim_size = max_len
        # #trim_size = 33
        # Trim = TrimSentences(trim_size)
        # X_train_pad = Trim.trim_list(X_train_pad)
        # print("X[0], ", X_train_pad[0])
        # #convert to numPY arrays
        X_train = np.array(edata)
        Y_train = np.array(Y_train)
        ones_count = np.count_nonzero(Y_train)
        zeros_count = len(Y_train) - ones_count
        print("ones count: ", ones_count)
        print("zeros count: ", zeros_count)
        print("two count: ", two_count)
        Y_train_old = Y_train
        Y_train = to_categorical(Y_train, num_classes=3)

        #Divide the data
        X_test_text = X_all[limit:]
        X_test = X_train[limit:]
        Y_test = Y_train[limit:]
        X_train = X_train[0: limit]
        Y_train = Y_train[0: limit]
        print("Entiendo que esto es la data que utiliza para hacer le training ", X_train)
        # print ("data divided on value: ", limit)
        # print("lengths X_train, Y_train: ", len(X_train), len(Y_train))
        # print("lengths X_test, Y_test: ", len(X_test), len(Y_test))

        print("Train data convert to numpy arrays")
        NN = KerasInceptionCNN(0, G)

        print("model created")
        kernel_regularizer = l2(0.001)

        NN.build(filters=11, first_dropout=0, second_dropout=0.05, padding='valid', dense_units=16)

        print("model built")
        NN.summary()

        sgd = SGD(lr=0.03, momentum=0.009, decay=0.001, nesterov=True)
        rmsprop = RMSprop(decay=0.003)
        adam = Adam(lr=0.1, decay=0.05)
        sgd = SGD(lr=0.05)


        NN.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy', precision, recall, f1, fprate])
        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        #class_weight = {0: 0.67, 1: 0.33}
        #class_weight = None
        history = NN.fit(X_train, Y_train, epochs=epochs, batch_size=32, callbacks=[callback], class_weight=class_weight_dictionary)
        print("Model trained")
        print("Predicting")
        print("len(X_test): ", X_test)
        preds = NN.predict(X_test)
        print("len(preds): ", len(preds))
        print("type preds: ", type(preds))
        print("preds before: ", preds)
        preds = np.argmax(preds, axis=1)
        print("preds: ", preds)
        print("len(preds): ", len(preds))
        Y_test = Y_train_old[limit:]
        print("Y test: ", Y_test)
        c_matrix = confusion_matrix(Y_test, preds)
        print("matrix: ", c_matrix)
        print("Storing Errors: ")
        ErrorAnalysis.store_errors(X_test_text, Y_test, preds, "errorcnn.csv")
        print("Errors stored")
        print("Confusion matrix: ")
        prec_1, recall_1, f1_1, spec_1, t = calculate_cm_metrics(c_matrix, '')
        print("C1-> presicion, recall, F1: ", prec_1, recall_1, f1_1)

        #
        # X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        # print("Test data mapped")
        # X_test_pad = P.pad_list(X_test_indices)
        # print("Test data padded")
        # X_test = np.array(X_test_pad)
        # Y_test = np.array(Y_test)
        # print("Test data converted to numpy arrays")
        # loss, acc = NN.evaluate(X_test, Y_test, callbacks=[callback])
        # print("accuracy: ", acc)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        X_Predict_Final = Trim.trim_list(X_Predict_Final)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", np.argmax(NN.predict(X_Predict_Final)))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        if plot:
            print("Ploting")
            self.plot(history)
        print("Done!")

