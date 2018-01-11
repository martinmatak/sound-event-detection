import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from librosa import display

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm, trange
from sklearn.metrics import roc_curve, auc
# constants for plots

# plt.style.use('ggplot')
#
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 12
# plt.rcParams['axes.labelsize'] = 11
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 14
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 11
# plt.rcParams['figure.titlesize'] = 13
#

# loading files and ploting some graphs

# def load_sound_files(file_paths):
#     raw_sounds = []
#     for fp in file_paths:
#         fp = "UrbanSound8K/audio/fold1/" + fp
#         X, sr = librosa.load(fp)
#         raw_sounds.append(X)
#     return raw_sounds
#
#
# def plot_waves(sound_names, raw_sounds):
#     i = 1
#     fig = plt.figure(figsize=(25, 60), dpi=90)
#     for n, f in zip(sound_names, raw_sounds):
#         plt.subplot(10, 1, i)
#         librosa.display.waveplot(np.array(f), sr=22050)
#         plt.title(n.title())
#         i += 1
#     plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.915, fontsize=18)
#     plt.show()
#
#
# def plot_specgram(sound_names, raw_sounds):
#     i = 1
#     fig = plt.figure(figsize=(25, 60), dpi=90)
#     for n, f in zip(sound_names, raw_sounds):
#         plt.subplot(10, 1, i)
#         specgram(np.array(f), Fs=22050)
#         plt.title(n.title())
#         i += 1
#     plt.suptitle('Figure 2: Spectrogram', x=0.5, y=0.915, fontsize=18)
#     plt.show()
#
#
# def plot_log_power_specgram(sound_names, raw_sounds):
#     i = 1
#     fig = plt.figure(figsize=(25, 60), dpi=90)
#     for n, f in zip(sound_names, raw_sounds):
#         plt.subplot(10, 1, i)
#         D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)
#         librosa.display.specshow(D, x_axis='time', y_axis='log')
#         plt.title(n.title())
#         i += 1
#     plt.suptitle('Figure 3: Log power spectrogram', x=0.5, y=0.915, fontsize=18)
#     plt.show()
#
#
# sound_file_paths = ["57320-0-0-7.wav", "24074-1-0-3.wav", "15564-2-0-1.wav", "31323-3-0-1.wav", "46669-4-0-35.wav",
#                     "89948-5-0-0.wav", "40722-8-0-4.wav", "103074-7-3-2.wav", "106905-8-0-0.wav", "108041-9-0-4.wav"]
# sound_names = ["not wrench", "car horn", "children playing", "dog bark", "drilling", "engine idling",
#                "gun shot", "jackhammer", "siren", "street music"]
#
# raw_sounds = load_sound_files(sound_file_paths)


# plot_waves(sound_names, raw_sounds)
# plot_specgram(sound_names, raw_sounds)
# plot_log_power_specgram(sound_names, raw_sounds)


# extracting features

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(parent_dir, sub_dirs, file_ext='*.mp4'):
    features, labels = np.empty((0, 193)), np.empty(0)
    index = 0
    dir_index = 0
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split('/')[-1].split('_')[3].split('-')[0])
    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def plot_roc(y_dec, pred):
    fpr, tpr, thresholds = roc_curve(y_dec, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for the neural network')
    plt.legend(loc="lower right")
    plt.show()


parent_dir = './'

sub_dirs = ['dataset']
features, labels = parse_audio_files(parent_dir, sub_dirs, '*.mp4')

labels = one_hot_encode(labels)

train_test_split = np.random.rand(len(features)) < 0.70
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]

#########################################
# training neural network with tensorflow
#########################################

training_epochs = 6000
n_dim = features.shape[1]
n_classes = 2
n_hidden_units_one = 380
n_hidden_units_two = 400
learning_rate = 0.01
sd = 1 / np.sqrt(n_dim)

# initializing weights
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

init = tf.global_variables_initializer()

# cost function, prediction and accuracy

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# y_ is what our model thinks, Y is a real value (label)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# running the flow
saver = tf.train.Saver()
cost_history = np.empty(shape=[1], dtype=float)
y_true, y_pred = None, None
i = 0
with tf.Session() as sess:
    sess.run(init)
    for epoch in tqdm(trange(training_epochs)):
        _, cost = sess.run([optimizer, cost_function], feed_dict={X: train_x, Y: train_y})
        cost_history = np.append(cost_history, cost)
    y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y, 1))
    y_dec = sess.run(y_, feed_dict={X: test_x})
    y_dec = y_dec[:, 1] / y_dec[:, 0]
    save_path = saver.save(sess, "./model/model_wrench.ckpt")
    print("Model saved in file: %s" % save_path)

plot_roc(y_true, y_dec)

fig = plt.figure(figsize=(10, 8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0, training_epochs, 0, np.max(cost_history)])
plt.show()

p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print("F-Score:", round(f, 3))
