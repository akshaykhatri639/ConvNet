import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ConvNet(object):
    """
    A convolutional network with the following architecture:

{conv-[batch norm]-relu-[dropout]-conv-[batch norm]-relu-[dropout]-pool}xN - {affine-[batch norm]-relu-[dropout]}xM - affine -[softmax]

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dims=[100], num_classes=10, weight_scale=1e-3, reg=0.0, 
               conv_layers=1, affine_layers=1, use_batchnorm = True, dropout=0,
               xavier=False, dtype=np.float32, seed=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Array of length affine_layers with number of units to use in each fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - conv_layers: no of instances of the [conv-relu-conv-relu-pool] layer
        - use_batchnorm and dropout self explanatory
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.filter_size = filter_size
        self.dtype = dtype
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        ############################################################################
        # TODO: Initialize weights and biases. Weights are initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
  
        w_count = 1
        prev_depth = input_dim[0]
        width = filter_size
        height = filter_size
        
        for i in range(conv_layers):
            if xavier:
                self.params['W' + str(w_count)] = np.random.randn(num_filters, prev_depth, height, width) / np.sqrt(prev_depth*height* width/2.0)
            else:
                self.params['W' + str(w_count)] = weight_scale * np.random.randn(num_filters, prev_depth, height, width)
            self.params['b' + str(w_count)] = np.zeros(num_filters)
            
            if use_batchnorm:
                key = 'bn' + str(w_count)
                bn = np.ones((2, num_filters))
                bn[1,:] = 0
                self.params[key] = bn
            
            w_count += 1
            prev_depth = num_filters
            if xavier:
                self.params['W' + str(w_count)] = np.random.randn(num_filters, prev_depth, height, width) / np.sqrt(prev_depth*height* width/2.0)
            else:
                self.params['W' + str(w_count)] = weight_scale * np.random.randn(num_filters, prev_depth, height, width)
            self.params['b' + str(w_count)] = np.zeros(num_filters)
            
            if use_batchnorm:
                key = 'bn' + str(w_count)
                bn = np.ones((2, num_filters))
                bn[1,:] = 0
                self.params[key] = bn
            
            w_count += 1
            prev_depth = num_filters
            Hout = 1 + (input_dim[1] + 2 * conv_param['pad'] - filter_size) / conv_param['stride'] 
            Wout = 1 + (input_dim[2] + 2 * conv_param['pad'] - filter_size) / conv_param['stride'] 
            height = 1 + (Hout - pool_param['pool_height']) / pool_param['stride'] 
            width = 1 + (Wout - pool_param['pool_width']) / pool_param['stride']
            
            
        inp = num_filters*height*width
        for i in range(affine_layers):
            if xavier:
                self.params['W' + str(w_count)] = np.random.randn(inp, hidden_dims[i])/np.sqrt(inp/2.0)
            else:
                self.params['W' + str(w_count)] = weight_scale * np.random.randn(inp, hidden_dims[i])
            self.params['b' + str(w_count)] = np.zeros(hidden_dims[i])
            
            if use_batchnorm:
                key = 'bn' + str(w_count)
                bn = np.ones((2, hidden_dims[i]))
                bn[1,:] = 0
                self.params[key] = bn
            
            w_count += 1
            inp = hidden_dims[i]
            
        if xavier:
            self.params['W' + str(w_count)] = np.random.randn(inp, num_classes)/np.sqrt(inp/2.0)
        else:
            self.params['W' + str(w_count)] = weight_scale * np.random.randn(inp, num_classes)
        self.params['b' + str(w_count)] = np.zeros(num_classes)
        self.w_count = w_count + 1
        self.conv_layers = conv_layers
        self.affine_layers = affine_layers
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(1, self.w_count - 1)]

        
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        mode = 'test' if y is None else 'train'
        w_count = self.w_count
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode 
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(w_count - 1)]

        
        conv_layers = self.conv_layers
        affine_layers = self.affine_layers 
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.filter_size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the convolutional net,              #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        N = X.shape[0]
        cache = {}
        layers = {}
        layers[1] = X
        wi = 1
        for i in range(conv_layers):
            cache_local = {}
            if self.use_batchnorm:
                layer, cache_local['conv'] = conv_bn_relu_fwd(layers[wi], self.params['W' + str(wi)], self.params['b' + str(wi)],                                                       self.params['bn' + str(wi)], conv_param, self.bn_params[wi])
            else:
                layer, cache_local['conv'] = conv_relu_forward(layers[wi], self.params['W' + str(wi)], self.params['b' + str(wi)],                                                       conv_param)
            
            if self.use_dropout:
                p = self.dropout_param['p']
                layer, cache_local['drop'] = dropout_forward(layer, self.dropout_param)

            cache[wi] = cache_local
            wi += 1
            layers[wi] = layer
            
            cache_local = {}
            if self.use_batchnorm:
                layer, cache_local['conv'] = conv_bn_relu_pool_fwd(layers[wi], self.params['W' + str(wi)], self.params['b' + str(wi)],                                                   self.params['bn' + str(wi)], conv_param, self.bn_params[wi], pool_param)
            else:
                layer, cache_local['conv'] = conv_relu_pool_forward(layers[wi], self.params['W' + str(wi)], self.params['b' + str(wi)],                                                 conv_param, pool_param)
            
            if self.use_dropout:
                p = self.dropout_param['p']
                layer, cache_local['drop'] = dropout_forward(layer, self.dropout_param)

            cache[wi] = cache_local
            wi += 1
            layers[wi] = layer
            
        for i in range(affine_layers):
            cache_local = {}
            if self.use_batchnorm:
                layer, cache_local['aff'] = affine_bn_relu_forward(layers[wi], self.params['W' + str(wi)], self.params['b' + str(wi)],                                                 self.params['bn' + str(wi)], self.bn_params[wi])
            else:
                layer, cache_local['aff'] = affine_relu_forward(layers[wi], self.params['W' + str(wi)], self.params['b' + str(wi)])
            
            if self.use_dropout:
                p = self.dropout_param['p']
                layer, cache_local['drop'] = dropout_forward(layer, self.dropout_param)

            cache[wi] = cache_local
            wi += 1
            layers[wi] = layer
        
        scores, cache[-1] = affine_forward(layers[wi], self.params['W' + str(wi)], self.params['b' + str(wi)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dl = softmax_loss(scores, y)
        upper_gradient = dl
        upper_gradient, grads['W' + str(wi)], grads['b' + str(wi)] = affine_backward(upper_gradient, cache[-1])
        loss += 0.5 * self.reg* np.sum(np.square(self.params['W' + str(wi)]))
        grads['W' + str(wi)] += self.reg*self.params['W' + str(wi)]
        wi -= 1
        
        for i in range(affine_layers):
            cache_local = cache[wi]
            loss += 0.5 * self.reg* np.sum(np.square(self.params['W' + str(wi)]))
            if self.use_dropout:
                upper_gradient = dropout_backward(upper_gradient, cache_local['drop'])
            if self.use_batchnorm:
                upper_gradient, grads['W' + str(wi)], grads['b' + str(wi)], grads['bn' + str(wi)] = affine_bn_relu_backward(                                                                                                   upper_gradient, cache_local['aff'])
            else:
                upper_gradient, grads['W' + str(wi)], grads['b' + str(wi)] = affine_relu_backward(upper_gradient, cache_local['aff'])
            
            grads['W' + str(wi)] += self.reg*self.params['W' + str(wi)]
            wi -= 1
            
        for i in range(conv_layers):
            cache_local = cache[wi]
            loss += 0.5 * self.reg* np.sum(np.square(self.params['W' + str(wi)]))
            if self.use_dropout:
                upper_gradient = dropout_backward(upper_gradient, cache_local['drop'])
            if self.use_batchnorm:
                upper_gradient, grads['W' + str(wi)], grads['b' + str(wi)], grads['bn' + str(wi)] = conv_bn_relu_pool_bkwd(                                                                                                   upper_gradient, cache_local["conv"])
            else:
                upper_gradient, grads['W' + str(wi)], grads['b' + str(wi)] = conv_relu_pool_backward(upper_gradient, cache_local['conv'])
            
            grads['W' + str(wi)] += self.reg*self.params['W' + str(wi)]
            wi -= 1
            
            cache_local = cache[wi]
            loss += 0.5 * self.reg* np.sum(np.square(self.params['W' + str(wi)]))
            if self.use_dropout:
                upper_gradient = dropout_backward(upper_gradient, cache_local['drop'])
            if self.use_batchnorm:
                upper_gradient, grads['W' + str(wi)], grads['b' + str(wi)], grads['bn' + str(wi)] = conv_bn_relu_bkwd( upper_gradient,                                                                                                         cache_local["conv"])
            else:
                upper_gradient, grads['W' + str(wi)], grads['b' + str(wi)] = conv_relu_backward(upper_gradient, cache_local['conv'])
            
            grads['W' + str(wi)] += self.reg*self.params['W' + str(wi)]
            wi -= 1
            
        return loss, grads