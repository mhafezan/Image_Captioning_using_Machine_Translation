import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from joblib import dump, load

"""We need to import several things from Keras."""

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def cache(cache_path, fn, *args, **kwargs):
    """
    Custom caching function to save and load precomputed data.
    If the cache file exists, it loads and returns it. Otherwise, it computes, saves, and returns the data.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        return load(cache_path)

    print(f"Processing and caching data at {cache_path}...")
    result = fn(*args, **kwargs)  # Compute the result
    dump(result, cache_path)  # Save the result
    return result

"""To load Flickr_8k_Dataset"""

# Define the dataset path
Flickr8k_Path = "D:/University of Lakehead/Deep Learning/Assignment_3/Flickr_8k_Dataset"
images_path = os.path.join(Flickr8k_Path, "Images")
captions_path = "D:/University of Lakehead/Deep Learning/Assignment_3/Flickr_8k_Dataset/captions.csv"

# Load captions
def load_flickr8k_captions(captions_file):
    """ Load captions from the Flickr8k dataset """

    # Read the CSV file (comma-separated)
    captions_data = pd.read_csv(captions_file)

    # Ensure correct column names (existing names are "image" and "caption")
    captions_data.rename(columns={"image": "filename"}, inplace=True)

    return captions_data

# Load dataset
captions_data = load_flickr8k_captions(captions_path)

# Get unique filenames
image_filenames = captions_data["filename"].unique()

# Split into training and validation sets
filenames_train, filenames_val = train_test_split(image_filenames, test_size=0.2, random_state=42)

# Get captions corresponding to the filenames
captions_train = captions_data[captions_data["filename"].isin(filenames_train)]
captions_val = captions_data[captions_data["filename"].isin(filenames_val)]

# Display dataset size
num_images_train = len(filenames_train)
num_images_val = len(filenames_val)

print(f"Training images: {num_images_train}, Validation images: {num_images_val}")

def load_image(path, size=None):
    """
    Load the image from the given file-path and resize it to the given size if not None.
    """

    # Load the image using PIL.
    img = Image.open(path)

    # Resize image if desired.
    if size is not None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    # Convert image to numpy array.
    img = np.array(img)

    # Normalize image-pixels so they fall between 0.0 and 1.0
    img = img / 255.0

    # Convert 2-dim grayscale image to 3-dim RGB
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    return img

def show_image(idx, train=True):
    """
    Load and plot an image from the Flickr8k training or validation set with the given index.
    """

    if train:
        if idx >= len(filenames_train):
            print("Index out of range for training set.")
            return
        filename = filenames_train[idx]
        captions = captions_train[captions_train["filename"] == filename]["caption"].tolist()
    else:
        if idx >= len(filenames_val):
            print("Index out of range for validation set.")
            return
        filename = filenames_val[idx]
        captions = captions_val[captions_val["filename"] == filename]["caption"].tolist()

    # Path for the image-file.
    path = os.path.join(images_path, filename)

    # Print the captions for this image.
    print(f"Image: {filename}")
    for caption in captions:
        print(f"- {caption}")

    # Load the image and plot it.
    img = load_image(path)
    plt.imshow(img)
    plt.axis("off")  # Hide axis for a cleaner image display
    plt.show()

"""Show an example image and captions from the training-set."""

show_image(idx=1, train=True)

# Pre-Trained Image Model (VGG16)

image_model = VGG16(include_top=True, weights='imagenet')

"""Print a list of all the layers in the VGG16 model."""

image_model.summary()

"""We will use the output of the layer prior to the final classification-layer which is named `fc2`. This is a fully-connected (or dense) layer."""

transfer_layer = image_model.get_layer('fc2')
transfer_layer.output.shape[1]

"""We call it the "transfer-layer" because we will transfer its output to another model that creates the image captions.

To do this, first we need to create a new model which has the same input as the original VGG16 model but outputs the transfer-values from the `fc2` layer.
"""

image_model_transfer = Model(inputs=image_model.input, outputs=transfer_layer.output)

"""The model expects input images to be of this size:"""

img_size = K.int_shape(image_model.input)[1:3]
img_size

"""For each input image, the new model will output a vector of transfer-values with this length:"""

transfer_values_size = K.int_shape(transfer_layer.output)[1]
transfer_values_size

# Process All Images

def print_progress(count, max_count):
    # Percentage completion.
    pct_complete = count / max_count

    # Status-message. Note the \r which means the line should
    # overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

"""This is the function for processing the given files using the VGG16-model and returning their transfer-values."""

def process_images(data_dir, filenames, batch_size=32):
    """
    Process all the given files in the given data_dir using the
    pre-trained image-model and return their transfer-values.

    Note that we process the images in batches to save
    memory and improve efficiency on the GPU.
    """

    # Number of images to process.
    num_images = len(filenames)

    # Pre-allocate input-batch-array for images.
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)

    # Pre-allocate output-array for transfer-values.
    # Note that we use 16-bit floating-points to save memory.
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    # Initialize index into the filenames.
    start_index = 0

    # Process batches of image-files.
    while start_index < num_images:
        # Print the percentage-progress.
        print_progress(count=start_index, max_count=num_images)

        # End-index for this batch.
        end_index = start_index + batch_size

        # Ensure end-index is within bounds.
        if end_index > num_images:
            end_index = num_images

        # The last batch may have a different batch-size.
        current_batch_size = end_index - start_index

        # Load all the images in the batch.
        for i, filename in enumerate(filenames[start_index:end_index]):
            # Path for the image-file.
            path = os.path.join(data_dir, filename)

            # Load and resize the image.
            # This returns the image as a numpy-array.
            img = load_image(path, size=img_size)

            # Save the image for later use.
            image_batch[i] = img

        # Use the pre-trained image-model to process the image.
        # Note that the last batch may have a different size,
        # so we only use the relevant images.
        transfer_values_batch = image_model_transfer.predict(image_batch[0:current_batch_size])

        # Save the transfer-values in the pre-allocated array.
        transfer_values[start_index:end_index] = transfer_values_batch[0:current_batch_size]

        # Increase the index for the next loop-iteration.
        start_index = end_index

    # Print newline.
    print()

    return transfer_values

"""Helper-function for processing all images in the training-set. This saves the transfer-values in a cache-file for fast reloading."""

def process_images_train():
    print("Processing {0} images in training-set ...".format(len(filenames_train)))

    # Path for saving the cache-file
    cache_path = os.path.join("D:/University of Lakehead/Deep Learning/Assignment_3/Flickr_8k_Dataset", "transfer_values_train.pkl")

    # Process images and save their transfer-values using Flickr8k dataset
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir="D:/University of Lakehead/Deep Learning/Assignment_3/Flickr_8k_Dataset/Images",
                            filenames=filenames_train)

    return transfer_values

"""Helper-function for processing all images in the validation-set."""

def process_images_val():
    print("Processing {0} images in validation-set ...".format(len(filenames_val)))

    # Path for the cache-file.
    cache_path = os.path.join("D:/University of Lakehead/Deep Learning/Assignment_3/Flickr_8k_Dataset", "transfer_values_val.pkl")

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir="D:/University of Lakehead/Deep Learning/Assignment_3/Flickr_8k_Dataset/Images",
                            filenames=filenames_val)

    return transfer_values

"""Process all images in the training-set and save the transfer-values to a cache-file. This took about 30 minutes to process on a GTX 1070 GPU."""

# Commented out IPython magic to ensure Python compatibility.
transfer_values_train = process_images_train()
print("dtype:", transfer_values_train.dtype)
print("shape:", transfer_values_train.shape)

"""Process all images in the validation-set and save the transfer-values to a cache-file. This took about 90 seconds to process on a GTX 1070 GPU."""

# Commented out IPython magic to ensure Python compatibility.
transfer_values_val = process_images_val()
print("dtype:", transfer_values_val.dtype)
print("shape:", transfer_values_val.shape)

# Tokenizer

mark_start = 'ssss '
mark_end = ' eeee'

"""This helper-function wraps all text-strings in the above markers. Note that the captions are a list of list, so we need a nested for-loop to process it. This can be done using so-called list-comprehension in Python."""

def mark_captions(captions_listlist):
    captions_marked = [[mark_start + caption + mark_end
                        for caption in captions_list]
                        for captions_list in captions_listlist]

    return captions_marked

# To convert captions_train to a list. It is Pandas Dataframe Initially.
captions_train = captions_train.groupby("filename")["caption"].apply(list).tolist()

"""Now process all the captions in the training-set and show an example."""

captions_train_marked = mark_captions(captions_train)
captions_train_marked[0]

"""This is how the captions look without the start- and end-markers."""

# captions_train is a pandas dataframe not a list
captions_train[0]

"""This helper-function converts a list-of-list to a flattened list of captions."""

def flatten(captions_listlist):
    captions_list = [caption
                     for captions_list in captions_listlist
                     for caption in captions_list]

    return captions_list

"""Now use the function to convert all the marked captions from the training set."""

captions_train_flat = flatten(captions_train_marked)

"""Set the maximum number of words in our vocabulary. This means that we will only use e.g. the 10000 most frequent words in the captions from the training-data."""

num_words = 10000

"""We need a few more functions than provided by Keras' Tokenizer-class so we wrap it."""

class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""

    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text

    def captions_to_tokens(self, captions_listlist):
        """
        Convert a list-of-list with text-captions to
        a list-of-list of integer-tokens.
        """

        # Note that text_to_sequences() takes a list of texts.
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]

        return tokens

"""Now create a tokenizer using all the captions in the training-data. Note that we use the flattened list of captions to create the tokenizer because it cannot take a list-of-lists."""

# Commented out IPython magic to ensure Python compatibility.
tokenizer = TokenizerWrap(texts=captions_train_flat, num_words=num_words)

"""Get the integer-token for the start-marker (the word "ssss"). We will need this further below."""

token_start = tokenizer.word_index[mark_start.strip()]
token_start

"""Get the integer-token for the end-marker (the word "eeee")."""

token_end = tokenizer.word_index[mark_end.strip()]
token_end

"""Convert all the captions from the training-set to sequences of integer-tokens. We get a list-of-list as a result."""

# Commented out IPython magic to ensure Python compatibility.
tokens_train = tokenizer.captions_to_tokens(captions_train_marked)

"""Example of the integer-tokens for the captions of the first image in the training-set:"""

print(tokens_train[0])

"""These are the corresponding text-captions:"""

captions_train_marked[0]

# Data Generator
def get_random_caption_tokens(idx):
    """
    Given a list of indices for images in the training-set,
    select a token-sequence for a random caption,
    and return a list of all these token-sequences.
    """

    # Initialize an empty list for the results.
    result = []

    # For each of the indices.
    for i in idx:
        # The index i points to an image in the training-set.
        # Each image in the training-set has at least 5 captions
        # which have been converted to tokens in tokens_train.
        # We want to select one of these token-sequences at random.

        # Get a random index for a token-sequence.
        j = np.random.choice(len(tokens_train[i]))

        # Get the j'th token-sequence for image i.
        tokens = tokens_train[i][j]

        # Add this token-sequence to the list of results.
        result.append(tokens)

    return result

"""This generator function creates random batches of training-data for use in training the neural network."""

def batch_generator(batch_size):
    """
    Generator function for creating random batches of training-data.
    """
    while True:
        # Get a list of random indices for images in the training-set.
        idx = np.random.randint(num_images_train, size=batch_size)

        # Get the pre-computed transfer-values for those images.
        transfer_values = transfer_values_train[idx]

        # Get random captions
        tokens = get_random_caption_tokens(idx)

        # Count the number of tokens in all these token-sequences.
        num_tokens = [len(t) for t in tokens]
        max_tokens = np.max(num_tokens)

        # Pad all the token-sequences with zeros
        tokens_padded = pad_sequences(tokens, maxlen=max_tokens, padding='post', truncating='post')

        # Prepare input and output sequences
        decoder_input_data = tokens_padded[:, 0:-1]
        # Ensure output data is int32 for sparse_categorical_crossentropy
        decoder_output_data = np.array(tokens_padded[:, 1:], dtype=np.int32)

        # Create input and output dictionaries with matching names
        x_data = {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transfer_values
        }

        y_data = {
            'decoder_output': decoder_output_data.reshape(decoder_output_data.shape + (1,))  # Add extra dimension for sparse categorical crossentropy
        }

        yield (x_data, y_data)

"""Set the batch-size used during training. This is set very high so the GPU can be used maximally - but this also requires a lot of RAM on the GPU. You may have to lower this number if the training runs out of memory."""

batch_size = 384

"""Create an instance of the data-generator."""
generator = batch_generator(batch_size=batch_size)


"""Test the data-generator by creating a batch of data."""
batch = next(generator)
batch_x = batch[0]
batch_y = batch[1]

"""Example of the transfer-values for the first image in the batch."""

batch_x['transfer_values_input'][0]

"""Example of the token-sequence for the first image in the batch. This is the input to the decoder-part of the neural network."""

batch_x['decoder_input'][0]

"""This is the token-sequence for the output of the decoder. Note how it is the same as the sequence above, except it is shifted one time-step."""

batch_y['decoder_output'][0]

# Steps Per Epoch

num_captions_train = [len(captions) for captions in captions_train]
num_captions_train[0]

"""This is the total number of captions in the training-set."""

total_num_captions_train = np.sum(num_captions_train)
total_num_captions_train

"""This is the approximate number of batches required per epoch, if we want to process each caption and image pair once per epoch."""

steps_per_epoch = int(total_num_captions_train / batch_size)
steps_per_epoch

# Implement the Recurrent Neural Network

state_size = 512

"""The embedding-layer converts integer-tokens into vectors of this length:"""

embedding_size = 128

"""This inputs transfer-values to the decoder:"""

transfer_values_input = Input(shape=(transfer_values_size,), name='transfer_values_input')

"""We want to use the transfer-values to initialize the internal states of the GRU units. This informs the GRU units of the contents of the images. The transfer-values are vectors of length 4096 but the size of the internal states of the GRU units are only 512, so we use a fully-connected layer to map the vectors from 4096 to 512 elements.

Note that we use a `tanh` activation function to limit the output of the mapping between -1 and 1, otherwise this does not seem to work.
"""

decoder_transfer_map = Dense(state_size, activation='tanh', name='decoder_transfer_map')

"""This is the input for token-sequences to the decoder. Using `None` in the shape means that the token-sequences can have arbitrary lengths."""

decoder_input = Input(shape=(None, ), name='decoder_input')

"""This is the embedding-layer which converts sequences of integer-tokens to sequences of vectors."""

decoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_size, name='decoder_embedding')

"""This creates the 3 GRU layers of the decoder. Note that they all return sequences because we ultimately want to output a sequence of integer-tokens that can be converted into a text-sequence."""

decoder_gru1 = GRU(state_size, name='decoder_gru1', return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2', return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3', return_sequences=True)

"""The GRU layers output a tensor with shape `[batch_size, sequence_length, state_size]`, where each "word" is encoded as a vector of length `state_size`. We need to convert this into sequences of integer-tokens that can be interpreted as words from our vocabulary.

One way of doing this is to convert the GRU output to a one-hot encoded array. It works but it is extremely wasteful, because for a vocabulary of e.g. 10000 words we need a vector with 10000 elements, so we can select the index of the highest element to be the integer-token.
"""

decoder_dense = Dense(num_words, activation='softmax', name='decoder_output')

""" Connect and Create the Training Model

The decoder is built using the functional API of Keras, which allows more flexibility in connecting the layers e.g. to have multiple inputs. This is useful e.g. if you want to connect the image-model directly with the decoder instead of using pre-calculated transfer-values.

This function connects all the layers of the decoder to some input of transfer-values.
"""

def connect_decoder(transfer_values):
    # Map the transfer-values so the dimensionality matches
    # the internal state of the GRU layers. This means
    # we can use the mapped transfer-values as the initial state
    # of the GRU layers.
    initial_state = decoder_transfer_map(transfer_values)

    # Start the decoder-network with its input-layer.
    net = decoder_input

    # Connect the embedding-layer.
    net = decoder_embedding(net)

    # Connect all the GRU layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)

    return decoder_output

"""Connect and create the model used for training. This takes as input transfer-values and sequences of integer-tokens and outputs sequences of one-hot encoded arrays that can be converted into integer-tokens."""

decoder_output = connect_decoder(transfer_values=transfer_values_input)

# Create the model with explicitly named outputs
decoder_model = Model(
    inputs=[transfer_values_input, decoder_input],
    outputs={'decoder_output': decoder_output}  # Name the output explicitly
)

# Compile the model with matching loss function
decoder_model.compile(
    optimizer=RMSprop(learning_rate=1e-3),
    loss={'decoder_output': 'sparse_categorical_crossentropy'}
)

""" Callback Functions

During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.

This is the callback for writing checkpoints during training.
"""

path_checkpoint = '22_checkpoint.weights.h5'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, verbose=1, save_weights_only=True)

"""This is the callback for writing the TensorBoard log during training."""

callback_tensorboard = TensorBoard(log_dir='./22_logs/', histogram_freq=0, write_graph=False)

callbacks = [callback_checkpoint, callback_tensorboard]

# Load Checkpoint
"""
try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)
"""

""" Train the Model

Now we will train the decoder so it can map transfer-values from the image-model to sequences of integer-tokens for the captions of the images.

One epoch of training took about 7 minutes on a GTX 1070 GPU. You probably need to run 20 epochs or more during training.

Note that if we didn't use pre-computed transfer-values then each epoch would take maybe 40 minutes to run, because all the images would have to be processed by the VGG16 model as well.
"""

# Commented out IPython magic to ensure Python compatibility.
decoder_model.fit(x=generator, steps_per_epoch=steps_per_epoch, epochs=20, callbacks=callbacks)

""" Generate Captions

This function loads an image and generates a caption using the model we have trained.
"""

def generate_caption(image_path, max_tokens=30):
    """
    Generate a caption for the image in the given path.
    The caption is limited to the given number of tokens (words).
    """

    # Load and resize the image.
    image = load_image(image_path, size=img_size)

    # Expand the 3-dim numpy array to 4-dim
    # because the image-model expects a whole batch as input,
    # so we give it a batch with just one image.
    image_batch = np.expand_dims(image, axis=0)

    # Process the image with the pre-trained image-model
    # to get the transfer-values.
    transfer_values = image_model_transfer.predict(image_batch)

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int32)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.

        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # This is the sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    # Plot the image.
    plt.imshow(image)
    plt.show()

    # Print the predicted caption.
    print("Predicted caption:")
    print(output_text)
    print()

def generate_caption_Flickr(idx, train=False):
    """
    Generate a caption for an image in the COCO data-set.
    Use the image with the given index in either the
    training-set (train=True) or validation-set (train=False).
    """

    if train:
        # Use image and captions from the training-set.
        filename = filenames_train[idx]
        captions = captions_train[idx]
    else:
        # Use image and captions from the validation-set.
        filename = filenames_val[idx]
        captions = captions_val[idx]

    # Path for the image-file.
    path = os.path.join(images_path, filename)

    # Use the model to generate a caption of the image.
    generate_caption(image_path=path)

    # Print the true captions from the data-set.
    print("True captions:")
    for caption in captions:
        print(caption)

"""Try this on a picture from the training-set that the model has been trained on. In some cases the generated caption is actually better than the human-generated captions."""

generate_caption_Flickr(idx=1, train=True)

"""Here is another picture of giraffes from the training-set, so this image was also used during training of the model. But the model can't produce an accurate caption. Perhaps it needs more training, or perhaps another architecture for the Recurrent Neural Network?"""

generate_caption_Flickr(idx=10, train=True)

"""Here is a picture from the validation-set which was not used during training of the model. Sometimes the model can produce good captions for images it hasn't seen during training and sometimes it can't. Can you make a better model?"""

generate_caption_Flickr(idx=1, train=False)

""" Evaluate Model Using BLEU Scores """

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk
try:
    nltk.download('punkt')
except:
    pass

def evaluate_model_bleu():
    """
    Evaluate the model using BLEU scores on the validation set
    """
    print("Evaluating model using BLEU scores...")
    
    # Lists to store actual and predicted captions
    actual_captions = []
    predicted_captions = []
    
    # Evaluate on validation set (using a subset for speed)
    num_samples = min(100, len(filenames_val))  # Use at most 100 samples
    
    for idx in range(num_samples):
        # Get the image filename and its captions
        filename = filenames_val[idx]
        true_captions = captions_val[captions_val["filename"] == filename]["caption"].tolist()
        
        # Generate caption for this image
        image_path = os.path.join(images_path, filename)
        image = load_image(image_path, size=img_size)
        image_batch = np.expand_dims(image, axis=0)
        transfer_values = image_model_transfer.predict(image_batch)
        
        # Generate caption
        shape = (1, 30)  # max_tokens = 30
        decoder_input_data = np.zeros(shape=shape, dtype=np.int32)
        token_int = token_start
        predicted_caption = ''
        count_tokens = 0
        
        while token_int != token_end and count_tokens < 30:
            decoder_input_data[0, count_tokens] = token_int
            x_data = {
                'transfer_values_input': transfer_values,
                'decoder_input': decoder_input_data
            }
            decoder_output = decoder_model.predict(x_data)
            token_onehot = decoder_output['decoder_output'][0, count_tokens, :]
            token_int = np.argmax(token_onehot)
            sampled_word = tokenizer.token_to_word(token_int)
            predicted_caption += " " + sampled_word
            count_tokens += 1
        
        # Process captions for BLEU score calculation
        predicted_tokens = nltk.word_tokenize(predicted_caption.lower())
        reference_captions = [[nltk.word_tokenize(cap.lower())] for cap in true_captions]
        
        actual_captions.append(reference_captions)
        predicted_captions.append(predicted_tokens)
        
        if idx % 10 == 0:
            print(f"Processed {idx+1}/{num_samples} images")
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(actual_captions, predicted_captions, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual_captions, predicted_captions, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(actual_captions, predicted_captions, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(actual_captions, predicted_captions, weights=(0.25, 0.25, 0.25, 0.25))
    
    print("\nBLEU Scores:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")

def visualize_predictions(num_samples=5):
    """
    Generate and visualize captions for random images from the validation set
    """
    plt.figure(figsize=(20, 4*num_samples))
    
    # Random indices from validation set
    random_indices = np.random.choice(len(filenames_val), num_samples, replace=False)
    
    for i, idx in enumerate(random_indices):
        # Get image and true captions
        filename = filenames_val[idx]
        true_captions = captions_val[captions_val["filename"] == filename]["caption"].tolist()
        image_path = os.path.join(images_path, filename)
        
        # Generate caption
        image = load_image(image_path, size=img_size)
        image_batch = np.expand_dims(image, axis=0)
        transfer_values = image_model_transfer.predict(image_batch)
        
        shape = (1, 30)
        decoder_input_data = np.zeros(shape=shape, dtype=np.int32)
        token_int = token_start
        predicted_caption = ''
        count_tokens = 0
        
        while token_int != token_end and count_tokens < 30:
            decoder_input_data[0, count_tokens] = token_int
            x_data = {
                'transfer_values_input': transfer_values,
                'decoder_input': decoder_input_data
            }
            decoder_output = decoder_model.predict(x_data)
            token_onehot = decoder_output['decoder_output'][0, count_tokens, :]
            token_int = np.argmax(token_onehot)
            sampled_word = tokenizer.token_to_word(token_int)
            predicted_caption += " " + sampled_word
            count_tokens += 1
        
        # Plot
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Predicted: {predicted_caption}\nActual: {true_captions[0]}', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# After training, evaluate the model
print("\nEvaluating model performance...")
evaluate_model_bleu()

print("\nGenerating example predictions...")
visualize_predictions(num_samples=5)