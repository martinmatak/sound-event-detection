import tensorflow as tf
import numpy as np
import os
import librosa
import glob


def parse_audio_files(parent_dir, sub_dirs, file_ext='*.mp4'):
    features = np.empty((0, 193))
    videos_indexes = []
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            index_start_inclusive = fn.find('-') + 1
            index_end_exclusive = len(fn) - len(file_ext) + 1
            video_index = fn[index_start_inclusive:index_end_exclusive]
            videos_indexes.append(int(video_index))
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
    return np.array(features), videos_indexes


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def convert_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d" % (m, s)

if __name__ == "__main__":
    features, video_indexes = parse_audio_files('./', ['tmp'])
    n_dim = features.shape[1]

    n_classes = 2
    n_hidden_units_one = 280
    n_hidden_units_two = 300
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

    with tf.Session() as sess:
        # Restore model weights from previously saved model
        tf.train.Saver().restore(sess, "./model/model_wrench.ckpt")
        y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: features})
        tensor = [int(x) for x in str(y_pred)[1:-1].split(' ')]
        index = 0
        for x in tensor:
            if x == 1:
                video_part = video_indexes[index]
                wrench_time = convert_time(video_part * 3)
                print("time: ", wrench_time)
            index += 1
