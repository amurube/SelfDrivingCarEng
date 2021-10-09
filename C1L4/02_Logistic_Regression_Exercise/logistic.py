import tensorflow as tf
import numpy as np
from utils import check_softmax, check_ce, check_model, check_acc


def softmax(logits):
    """
    softmax implementation
    args:
    - logits [tensor]: 1xN logits tensor
    returns:
    - soft_logits [tensor]: softmax of logits
    """
    # IMPLEMENT THIS FUNCTION
    soft_logits = np.exp(logits)/((np.exp(logits)).sum())
    
    #alternative way
    exp = tf.exp(logits)
    denom = tf.reduce_sum(exp, 1, keepdims = True)
    soft_logits_alternative = exp / denom
    
    return soft_logits


def cross_entropy(scaled_logits, one_hot):
    """
    Cross entropy loss implementation
    args:
    - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
    - one_hot [tensor]: one hot tensor
    returns:
    - loss [tensor]: cross entropy 
    """
    # IMPLEMENT THIS FUNCTION
    nll = - np.sum(one_hot * np.log(scaled_logits))
    
    #alternative way
    masked_logits = tf.boolean_mask(scaled_logits, one_hot)
    nll_alternative = - tf.math.log(masked_logits)
    
    return nll


def model(X, W, b):
    """
    logistic regression model
    args:
    - X [tensor]: input HxWx3
    - W [tensor]: weights
    - b [tensor]: bias
    returns:
    - output [tensor]
    """
    # IMPLEMENT THIS FUNCTION
    y = softmax(tf.matmul(tf.reshape(X, (1, -1)),  W) + b)
    
    return y


def accuracy(y_hat, Y):
    """
    calculate accuracy
    args:
    - y_hat [tensor]: NxC tensor of models predictions
    - y [tensor]: N tensor of ground truth classes
    returns:
    - acc [tensor]: accuracy
    """
    # IMPLEMENT THIS FUNCTION
    # calculate argmax
    argmax = tf.cast(tf.argmax(y_hat, axis=1), Y.dtype)

    # calculate acc
    acc = tf.math.reduce_sum(tf.cast(argmax == Y, tf.int32)) / Y.shape[0]
    return acc


if __name__ == '__main__':

    check_softmax(softmax)
    
    # checking the NLL implementation
    check_ce(cross_entropy)
    
    check_model(model)

    check_acc(accuracy)