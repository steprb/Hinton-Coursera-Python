import numpy as np
import scipy.io as sio
from time import time
import pickle

def load_data(N):
    '''This method loads the training, validation and test set.
    It also divides the training set into mini-batches.
    Inputs:
        N: Mini-batch size.
    Outputs:
        train_input: An array of size D X N X M, where
            D: number of input dimensions (in this case, 3).
            N: size of each mini-batch (in this case, 100).
            M: number of minibatches.
        train_target: An array of size 1 X N X M.
        valid_input: An array of size D X number of points in the validation set.
        test: An array of size D X number of points in the test set.
        vocab: Vocabulary containing index to word mapping.
    '''

    # Load data
    data = sio.loadmat('data.mat')

    trainData = data['data'][0][0]['trainData']
    validData = data['data'][0][0]['validData']
    testData = data['data'][0][0]['testData']
    vocab = data['data'][0][0]['vocab']

    numdims = trainData.shape[0]
    D = numdims-1
    M = int(trainData.shape[1] / N)

    train_input = trainData[0:D,0:N*M].reshape(D, N, M, order="F")
    train_target = trainData[D,0:N*M].reshape(1, N, M, order="F")

    valid_input = validData[0:D]
    valid_target = validData[D].reshape(1,validData.shape[1], order="F")
    test_input = testData[0:D]
    test_target = testData[D].reshape(1,testData.shape[1], order="F")

    return [train_input, train_target, valid_input, valid_target,
            test_input, test_target, vocab]

def train(epochs):
    ''' This function trains a neural network language model.
    Inputs:
        epochs: Number of epochs to run.
    Output:
        model: A struct containing the learned weights and biases and vocabulary.
    '''

    start_time = time()

    # SET HYPERPARAMETERS HERE
    batchsize = 100 # Mini-batch size.
    learning_rate = 0.1 # Learning rate; default = 0.1.
    momentum = 0.9 # Momentum; default = 0.9.
    numhid1 = 50 # Dimensionality of embedding space; default = 50.
    numhid2 = 200 # Number of units in hidden layer; default = 200.
    init_wt = 0.01  # Standard deviation of the normal distribution
                    # which is sampled to get the initial weights; default = 0.01

    # VARIABLES FOR TRACKING TRAINING PROGRESS
    show_training_CE_after = 100
    show_validation_CE_after = 1000

    # LOAD DATA
    [train_input, train_target, valid_input, valid_target,
            test_input, test_target, vocab] = load_data(batchsize)

    # Decrease all word indices by 1: 1-250 -> 0-249
    train_input = train_input - 1
    train_target = train_target - 1
    valid_input = valid_input - 1
    valid_target = valid_target - 1
    test_input = test_input - 1
    test_target = test_target - 1

    [numwords, batchsize, numbatches] = train_input.shape
    vocab_size = vocab.shape[1]

    # INITIALIZE WEIGHTS AND BIASES
    word_embedding_weights = init_wt * np.random.randn(vocab_size, numhid1)
    embed_to_hid_weights = init_wt * np.random.randn(numwords * numhid1, numhid2)
    hid_to_output_weights = init_wt * np.random.randn(numhid2, vocab_size)
    hid_bias = np.zeros((numhid2,1))
    output_bias = np.zeros((vocab_size,1))

    word_embedding_weights_delta = np.zeros((vocab_size, numhid1))
    word_embedding_weights_gradient = np.zeros((vocab_size, numhid1))
    embed_to_hid_weights_delta = np.zeros((numwords * numhid1, numhid2))
    hid_to_output_weights_delta = np.zeros((numhid2, vocab_size))
    hid_bias_delta = np.zeros((numhid2, 1))
    output_bias_delta = np.zeros((vocab_size, 1))

    expansion_matrix = np.eye(vocab_size)
    count = 0
    tiny = np.exp(-30)

    # TRAIN
    for epoch in range(epochs):
        print("Epoch", epoch+1)
        this_chunk_CE = 0
        trainset_CE = 0

        # LOOP OVER MINI-BATCHES
        for m in range(numbatches):
            input_batch = train_input[:,:,m]
            target_batch = train_target[:,:,m]

            # FORWARD PROPAGATE
            # Compute the state of each layer in the network given the input batch
            # and all weights and biases
            [embedding_layer_state, hidden_layer_state, output_layer_state] = \
              fprop(input_batch, word_embedding_weights, embed_to_hid_weights,
                                hid_to_output_weights, hid_bias, output_bias)

            # COMPUTE DERIVATIVE
            # Expand the target to a sparse 1-of-K vector.
            expanded_target_batch = expansion_matrix[:,target_batch.ravel()]
            # Compute derivative of cross-entropy loss function.
            error_deriv = output_layer_state - expanded_target_batch

            # MEASURE LOSS FUNCTION
            CE = -sum(sum(expanded_target_batch*
                          np.log(output_layer_state+tiny)))/batchsize

            count =  count + 1
            this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count
            trainset_CE = trainset_CE + (CE - trainset_CE) / (m+1)

            if (m+1) % show_training_CE_after == 0:
                # In original octave version next line is outside of if.
                print("Batch", m+1, "Train this_chunk_CE", this_chunk_CE)
                count = 0
                this_chunk_CE = 0

            # BACK PROPAGATE
            # OUTPUT LAYER
            hid_to_output_weights_gradient = np.dot(hidden_layer_state,
                            np.transpose(error_deriv))
            output_bias_gradient = np.sum(error_deriv, axis=1)
            back_propagated_deriv_1 = np.dot(hid_to_output_weights, error_deriv)*\
                hidden_layer_state*(1-hidden_layer_state)

            # HIDDEN LAYER
            # FILL IN CODE. Replace the line below by one of the options.
            embed_to_hid_weights_gradient = np.zeros((numhid1 * numwords, numhid2))

            # Options:
            # (a)
            # embed_to_hid_weights_gradient = np.dot(
            #    np.transpose(back_propagated_deriv_1), embedding_layer_state)

            # (b)
            # embed_to_hid_weights_gradient = np.dot(embedding_layer_state,
            #                             np.transpose(back_propagated_deriv_1))

            # (c)
            # embed_to_hid_weights_gradient = back_propagated_deriv_1

            # (d)
            # embed_to_hid_weights_gradient = embedding_layer_state


            # FILL IN CODE. Replace the line below by one of the options.
            hid_bias_gradient = np.zeros((numhid2, 1))

            # Options
            # (a)
            # hid_bias_gradient = np.sum(back_propagated_deriv_1, axis=1)

            # (b)
            # hid_bias_gradient = np.sum(back_propagated_deriv_1, axis=0)

            # (c)
            # hid_bias_gradient = back_propagated_deriv_1

            # (d)
            # hid_bias_gradient = np.transpose(back_propagated_deriv_1)

            hid_bias_gradient = hid_bias_gradient.reshape(numhid2,-1, order="F")


            # FILL IN CODE. Replace the line below by one of the options.
            back_propagated_deriv_2 = np.zeros((numhid2, batchsize))

            # Options
            # (a)
            # back_propagated_deriv_2 = np.dot(embed_to_hid_weights,
            #                                     back_propagated_deriv_1)

            # (b)
            # back_propagated_deriv_2 = np.dot(back_propagated_deriv_1,
            #                                         embed_to_hid_weights)

            # (c)
            # back_propagated_deriv_2 = np.dot(
            #     np.transpose(back_propagated_deriv_1), embed_to_hid_weights)

            # (d)
            # back_propagated_deriv_2 = np.dot(back_propagated_deriv_1,
            #                           np.transpose(embed_to_hid_weights))


            word_embedding_weights_gradient.fill(0)

            # EMBEDDING LAYER
            for w in range(numwords):
                exp_tmp = expansion_matrix[:,input_batch[w,:]]
                bpd_tmp = back_propagated_deriv_2 \
                            [w * numhid1 : (w+1) * numhid1, :]
                word_embedding_weights_gradient = \
                        word_embedding_weights_gradient + \
                        np.dot(exp_tmp,np.transpose(bpd_tmp))

            # UPDATE WEIGHTS AND BIASES
            word_embedding_weights_delta = \
                momentum * word_embedding_weights_delta + \
                word_embedding_weights_gradient / batchsize
            word_embedding_weights = word_embedding_weights - \
                learning_rate * word_embedding_weights_delta

            embed_to_hid_weights_delta = \
                momentum * embed_to_hid_weights_delta + \
                embed_to_hid_weights_gradient / batchsize
            embed_to_hid_weights = embed_to_hid_weights - \
                learning_rate * embed_to_hid_weights_delta

            hid_to_output_weights_delta = \
                momentum * hid_to_output_weights_delta + \
                hid_to_output_weights_gradient / batchsize
            hid_to_output_weights = hid_to_output_weights - \
                learning_rate * hid_to_output_weights_delta

            hid_bias_delta = momentum * hid_bias_delta + \
                hid_bias_gradient / batchsize
            hid_bias = hid_bias - learning_rate * hid_bias_delta

            output_bias_delta = momentum * output_bias_delta + \
                output_bias_gradient.reshape(
                    output_bias_delta.shape[0],-1, order="F") / batchsize
            output_bias = output_bias - learning_rate * output_bias_delta

            # VALIDATE
            if m % show_validation_CE_after == 0:
                print("Running validation ...")
                [embedding_layer_state, hidden_layer_state,
                    output_layer_state]= fprop(valid_input,
                    word_embedding_weights, embed_to_hid_weights,
                    hid_to_output_weights, hid_bias, output_bias)

                datasetsize = valid_input.shape[1]
                expanded_valid_target = expansion_matrix[:, valid_target.ravel()]
                CE = -sum(sum(expanded_valid_target *
                              np.log(output_layer_state + tiny))) /datasetsize
                print("Validation CE:", CE)

        print("Average Training CE", trainset_CE)

    #### -----------------

    print("Finished Training")
    print("Final Training CE: ", trainset_CE)

    # EVALUATE ON VALIDATION SET
    print("Running validation...")

    [embedding_layer_state, hidden_layer_state, output_layer_state] = \
        fprop(valid_input, word_embedding_weights, embed_to_hid_weights,
            hid_to_output_weights, hid_bias, output_bias)
    datasetsize = valid_input.shape[1]
    expanded_valid_target = expansion_matrix[:, valid_target.ravel()]

    CE = -sum(sum( \
        expanded_valid_target * np.log(output_layer_state + tiny))) / datasetsize
    print("Final validation CE", CE)

    # EVALUATE ON TEST SET
    print("Running test...")

    [embedding_layer_state, hidden_layer_state, output_layer_state] = \
        fprop(test_input, word_embedding_weights, embed_to_hid_weights,
            hid_to_output_weights, hid_bias, output_bias)

    datasetsize = test_input.shape[1]
    expanded_test_target = expansion_matrix[:, test_target.ravel()]
    CE = -sum(sum( \
        expanded_test_target * np.log(output_layer_state + tiny))) / datasetsize
    print("Final Test CE", CE)

    model = {}
    model['word_embedding_weights'] = word_embedding_weights
    model['embed_to_hid_weights'] = embed_to_hid_weights
    model['hid_to_output_weights'] = hid_to_output_weights
    model['hid_bias'] = hid_bias
    model['output_bias'] = output_bias
    model['vocab'] = vocab

    end_time = time()
    print("Training took", end_time-start_time, "seconds")

    return model

def display_nearest_words(word, model, k):
    ''' Shows the k-nearest words to the query word.
    Inputs:
        word: The query word as a string.
        model: Model returned by the training script.
        k: The number of nearest words to display.
    Example usage:
        display_nearest_words('school', model, 10);
    '''

    word_embedding_weights = model['word_embedding_weights']
    vocab = model['vocab']

    vocab_list = []
    for i in np.nditer(vocab[0], flags=['refs_ok']):
        vocab_list.append(i.tolist()[0])

    if word in vocab_list:
        id = vocab_list.index(word)
    else:
        print("Word", word, "is not in vocabulary.")
        return

    # # Compute distance to every other word.
    vocab_size = len(vocab_list)
    word_rep = np.tile(word_embedding_weights[id,:],(vocab_size,1))
    diff = word_embedding_weights - word_rep
    dist = np.sqrt(np.sum(diff*diff,axis=1))

    # Sort by distance.
    dist_table = list(zip(dist.tolist(),vocab_list))
    sorted_table = sorted(dist_table, key=lambda tup: tup[0])

    for i in range(k):
        print(sorted_table[i+1])

def fprop(input_batch, word_embedding_weights, embed_to_hid_weights,
                        hid_to_output_weights, hid_bias, output_bias):
    '''
    This method forward propagates through a neural network.
    Inputs:
    input_batch: The input data as a matrix of size numwords X batchsize where,
        numwords is the number of words, batchsize is the number of data points.
        So, if input_batch(i, j) = k then the ith word in data point j is word
        index k of the vocabulary.

    word_embedding_weights: Word embedding as a matrix of size
        vocab_size X numhid1, where vocab_size is the size of the vocabulary
        numhid1 is the dimensionality of the embedding space.

    embed_to_hid_weights: Weights between the word embedding layer and hidden
        layer as a matrix of soze numhid1*numwords X numhid2, numhid2 is the
        number of hidden units.

    hid_to_output_weights: Weights between the hidden layer and output softmax
        unit as a matrix of size numhid2 X vocab_size

    hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.

    output_bias: Bias of the output layer as a matrix of size vocab_size X 1.

    Outputs:
    embedding_layer_state: State of units in the embedding layer as a matrix of
        size numhid1*numwords X batchsize

    hidden_layer_state: State of units in the hidden layer as a matrix of size
        numhid2 X batchsize

    output_layer_state: State of units in the output layer as a matrix of size
        vocab_size X batchsize

    '''

    [numwords, batchsize] = input_batch.shape
    [vocab_size, numhid1] = word_embedding_weights.shape
    numhid2 = embed_to_hid_weights.shape[1]

    # COMPUTE STATE OF WORD EMBEDDING LAYER
    # Look up the inputs word indices in the word_embedding_weights matrix.

    # Combine three inputs, turn them into 1D vector
    ib_tmp = np.reshape(input_batch, (1,-1), order="F").ravel()
    # Take rows from weight matrices, row number equal to word index.
    # Then transpose.
    wew_tmp = np.transpose(word_embedding_weights[ib_tmp])
    # Combine three nearby columns
    embedding_layer_state = np.reshape(wew_tmp, (numhid1 * numwords,-1), order="F")

    ## COMPUTE STATE OF HIDDEN LAYER
    ## Compute inputs to hidden units.
    inputs_to_hidden_units = np.dot(
            np.transpose(embed_to_hid_weights),embedding_layer_state)
    hid_bias_tmp = np.tile(hid_bias,(1,batchsize))
    inputs_to_hidden_units = inputs_to_hidden_units + hid_bias_tmp

    # Apply logistic activation function
    # FILL IN CODE. Replace the line below by one of the options.
    hidden_layer_state = np.zeros((numhid2, batchsize))

    # Options
    # (a)
    # hidden_layer_state = 1 / (1 + np.exp(inputs_to_hidden_units))

    # (b)
    # hidden_layer_state = 1 / (1 - np.exp(-inputs_to_hidden_units))

    # (c)
    # hidden_layer_state = 1 / (1 + np.exp(-inputs_to_hidden_units))

    # (d)
    # hidden_layer_state = -1 / (1 + np.exp(-inputs_to_hidden_units))


    # COMPUTE STATE OF OUTPUT LAYER
    # Compute inputs to softmax
    # FILL IN CODE. Replace the line below by one of the options.
    inputs_to_softmax = np.zeros((vocab_size, batchsize))

    # Options
    # (a)
    # inputs_to_softmax = np.dot(np.transpose(hid_to_output_weights),
    #                             hidden_layer_state)
    # output_bias_tmp = np.tile(output_bias,(1,batchsize))
    # inputs_to_softmax = inputs_to_softmax + output_bias_tmp

    # (b)
    # inputs_to_softmax = np.dot(np.transpose(hid_to_output_weights),
    #                            hidden_layer_state)
    # output_bias_tmp = np.tile(output_bias,(batchsize,1))
    # inputs_to_softmax = inputs_to_softmax + output_bias_tmp

    # (c)
    # inputs_to_softmax = np.dot(hidden_layer_state,
    #                            np.transpose(hid_to_output_weights))
    # output_bias_tmp = np.tile(output_bias,(1,batchsize))
    # inputs_to_softmax = inputs_to_softmax + output_bias_tmp

    # (d)
    # inputs_to_softmax = np.dot(hid_to_output_weights, hidden_layer_state)
    # output_bias_tmp = np.tile(output_bias,(batchsize,1))
    # inputs_to_softmax = inputs_to_softmax + output_bias_tmp


    # Subtract maximum
    # Remember that adding or subtracting the same constant from each input to a
    # softmax unit does not affect the outputs. Here we are subtracting maximum to
    # make all inputs <= 0. This prevents overflows when computing their
    # exponents.

    inputs_to_softmax = inputs_to_softmax - \
            np.tile(np.amax(inputs_to_softmax,axis=0),(vocab_size, 1))

    # Compute exp
    output_layer_state = np.exp(inputs_to_softmax)

    # Normalize to get probability distribution
    output_layer_state = output_layer_state / \
            np.tile(np.sum(output_layer_state,axis=0),(vocab_size,1))

    return [embedding_layer_state, hidden_layer_state, output_layer_state]

def predict_next_word(word1, word2, word3, model, k):
    ''' Predicts the next word.
    Inputs:
        word1: The first word as a string.
        word2: The second word as a string.
        word3: The third word as a string.
        model: Model returned by the training script.
        k: The k most probable predictions are shown.
    Example usage:
        predict_next_word('john', 'might', 'be', model, 3);
        predict_next_word('life', 'in', 'new', model, 3);
    '''

    vocab = model['vocab']

    vocab_list = []

    for i in np.nditer(vocab[0], flags=['refs_ok']):
        vocab_list.append(i.tolist()[0])

    if word1 in vocab_list:
        id1 = vocab_list.index(word1)
    else:
        print("Word", word1, "is not in vocabulary.")
        return

    if word2 in vocab_list:
        id2 = vocab_list.index(word2)
    else:
        print("Word", word2, "is not in vocabulary.")
        return

    if word3 in vocab_list:
        id3 = vocab_list.index(word3)
    else:
        print("Word", word3, "is not in vocabulary.")
        return

    input = [id1, id2, id3]

    input_array = np.asarray(input).reshape((-1,1))

    [embedding_layer_state, hidden_layer_state, output_layer_state] = \
      fprop(input_array, model['word_embedding_weights'],
            model['embed_to_hid_weights'], model['hid_to_output_weights'],
            model['hid_bias'], model['output_bias'])

    prob_table = list(zip(output_layer_state.tolist(),vocab_list))
    sorted_table = sorted(prob_table, key=lambda tup: tup[0], reverse=True)

    print(word1, word2, word3)
    for i in range(k):
        print(sorted_table[i])

def word_distance(word1, word2, model):
    ''' Shows the L2 distance between word1 and word2 in the
        word_embedding_weights.
    Inputs:
        word1: The first word as a string.
        word2: The second word as a string.
        model: Model returned by the training script.
    Example usage:
        word_distance('school', 'university', model)
    '''

    word_embedding_weights = model['word_embedding_weights']
    vocab = model['vocab']

    vocab_list = []
    for i in np.nditer(vocab[0], flags=['refs_ok']):
        vocab_list.append(i.tolist()[0])

    if word1 in vocab_list:
        id1 = vocab_list.index(word1)
    else:
        print("Word", word1, "is not in vocabulary.")
        return

    if word2 in vocab_list:
        id2 = vocab_list.index(word2)
    else:
        print("Word", word2, "is not in vocabulary.")
        return

    word_rep1 = word_embedding_weights[id1, :]
    word_rep2 = word_embedding_weights[id2, :]
    diff = word_rep1 - word_rep2
    distance = np.sqrt(sum(diff * diff))

    return distance

#### Main program

[train_x, train_t, valid_x, valid_t, test_x, test_t, vocab] = load_data(100)
model = train(10)

# Uncomment to save trained model for later use
# pickle.dump(model, open("learned_model.pkl", "wb"))

# Uncomment to open previously saved trained model
# model = pickle.load(open("learned_model.pkl", "rb"))
