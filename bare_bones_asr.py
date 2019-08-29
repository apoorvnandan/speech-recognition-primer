import tensorflow as tf
import numpy as np
import librosa
from string import ascii_lowercase


class ASR(tf.keras.Model):
    '''
    Class for defining the end to end ASR model.
    This model consists of a 1D convolutional layer followed by a bidirectional LSTM
    followed by a fully connected layer applied at each timestep.
    This is a bare-bones architecture.
    Experiment with your own architectures to get a good WER
    '''
    def __init__(self, filters, kernel_size, conv_stride, conv_border, n_lstm_units, n_dense_units):
        super(ASR, self).__init__()
        self.conv_layer = tf.keras.layers.Conv1D(filters,
                                                 kernel_size,
                                                 strides=conv_stride,
                                                 padding=conv_border,
                                                 activation='relu')
        self.lstm_layer = tf.keras.layers.LSTM(n_lstm_units,
                                               return_sequences=True,
                                               activation='tanh')
        self.lstm_layer_back = tf.keras.layers.LSTM(n_lstm_units,
                                                    return_sequences=True,
                                                    go_backwards=True,
                                                    activation='tanh')
        self.blstm_layer = tf.keras.layers.Bidirectional(self.lstm_layer, backward_layer=self.lstm_layer_back)
        self.dense_layer = tf.keras.layers.Dense(n_dense_units)

    def call(self, x):
        x = self.conv_layer(x)
        x = self.blstm_layer(x)
        x = self.dense_layer(x)
        return x


def compute_ctc_loss(logits, labels, logit_length, label_length):
    '''
    function to compute CTC loss.
    Note: tf.nn.ctc_loss applies log softmax to its input automatically
    :param logits: Logits from the output dense layer
    :param labels: Labels converted to array of indices
    :param logit_length: Array containing length of each input in the batch
    :param label_length: Array containing length of each label in the batch
    :return: array of ctc loss for each element in batch
    '''
    return tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False,
        unique=None,
        blank_index=-1,
        name=None
    )


def create_spectrogram(signals):
    '''
    function to create spectrogram from signals loaded from an audio file
    :param signals:
    :return:
    '''
    stfts = tf.signal.stft(signals, frame_length=200, frame_step=80, fft_length=256)
    spectrograms = tf.math.pow(tf.abs(stfts), 0.5)
    return spectrograms


def generate_input_from_audio_file(path_to_audio_file, resample_to=8000):
    '''
    function to create input for our neural network from an audio file.
    The function loads the audio file using librosa, resamples it, and creates spectrogram form it
    :param path_to_audio_file: path to the audio file
    :param resample_to:
    :return: spectrogram corresponding to the input file
    '''
    # load the signals and resample them
    signal, sample_rate = librosa.core.load(path_to_audio_file)
    if signal.shape[0] == 2:
        signal = np.mean(signal, axis=0)
    signal_resampled = librosa.core.resample(signal, sample_rate, resample_to)

    # create spectrogram
    X = create_spectrogram(signal_resampled)

    # normalisation
    means = tf.math.reduce_mean(X, 1, keepdims=True)
    stddevs = tf.math.reduce_std(X, 1, keepdims=True)
    X = tf.divide(tf.subtract(X, means), stddevs)
    return X


def generate_target_output_from_text(target_text):
    '''
    Target output is an array of indices for each character in your string.
    The indices comes from a mapping that will
    be used while decoding the ctc output.
    :param target_text: (str) target string
    :return: array of indices for each character in the string
    '''
    space_token = ' '
    end_token = '>'
    blank_token = '%'
    alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]
    char_to_index = {}
    for idx, char in enumerate(alphabet):
        char_to_index[char] = idx

    y = []
    for char in target_text:
        y.append(char_to_index[char])
    return y


def train_sample(x, y, optimizer, model):
    '''
    function perform forward and backpropagation on one batch
    :param x: one batch of input
    :param y: one batch of target
    :param optimizer: optimizer
    :param model: object of the ASR class
    :return: loss from this step
    '''
    with tf.GradientTape() as tape:
        logits = model(x)
        labels = y
        logits_length = [logits.shape[1]]*logits.shape[0]
        labels_length = [labels.shape[1]]*labels.shape[0]
        loss = compute_ctc_loss(logits, labels, logit_length=logits_length, label_length=labels_length)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(model, optimizer, X, Y, epochs):
    '''
    function to train the model for given number of epochs
    Note:
    For this example, I am passing a single batch of input to this function
    Therefore, the loop for iterating through batches is missing
    :param model: object of class ASR
    :param optimizer: optimizer
    :param X:
    :param Y:
    :param epochs:
    :return: None
    '''
    for step in range(1, epochs):
        loss = train_sample(X, Y, optimizer, model)
        print('Epoch {}, Loss: {}'.format(step, loss))


if __name__ == '__main__':
    sample_call = 'sample.wav'
    transcript = 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'.lower()

    X = generate_input_from_audio_file(sample_call)
    X = tf.expand_dims(X, axis=0)  # converting input into a batch of size 1
    y = generate_target_output_from_text(transcript)
    y = tf.expand_dims(tf.convert_to_tensor(y), axis=0)  # converting output to a batch of size 1
    print('Input shape: {}'.format(X.shape))
    print('Target shape: {}'.format(y.shape))

    model = ASR(200, 11, 2, 'valid', 200, 29)
    optimizer = tf.keras.optimizers.Adam()
    train(model, optimizer, X, y, 100)

    # getting the ctc output
    ctc_output = model(X)
    ctc_output = tf.nn.log_softmax(ctc_output)

    # greedy decoding
    space_token = ' '
    end_token = '>'
    blank_token = '%'
    alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]
    output_text = ''
    for timestep in ctc_output[0]:
        output_text += alphabet[tf.math.argmax(timestep)]
    print(output_text)
    print('\n\nNote: Applying a good decoder on this output will give you readable output')
