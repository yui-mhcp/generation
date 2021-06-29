import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *

from custom_architectures.current_blocks import _get_var, Conv2DTransposeBN, DenseBN

def simple_generator(noise_size,
                     output_shape,
                     n_blocks    = 3,
                    
                     n_dense     = 0,
                     dense_sizes = [],
                     dense_bias  = True,
                     dense_activation    = 'leaky',
                     dense_drop_rate    = 0.,
                     final_dense_bias     = False,
                     final_dense_activation   = 'leaky',
                     final_dense_bnorm        = 'after',
                    
                     conv_input_channels = 128,
                    
                     filters         = lambda i: 128 // 2 ** i,
                     kernel_size     = 5,
                     strides         = lambda i: 1 if i == 0 else 2,
                     use_bias        = False,
                     activation      = 'leaky',
                     bnorm           = 'after',
                     drop_rate       = 0.5,
                    
                     n_conv          = 0,
                     conv_filters    = 32,
                     conv_kernel_size    = 5,
                     conv_bias       = False,
                     conv_dilation   = 1,
                     conv_activation = 'leaky',
                     conv_bnorm           = 'after',
                     conv_drop_rate       = 0.25,
                    
                     final_kernel_size   = 5,
                     final_strides       = 2,
                     final_bias          = False,
                     final_activation    = 'sigmoid',
                    
                     use_conv_as_final   = False,
                      
                     name = 'generator',
                     ** kwargs
                    ):
    w, _, output_channels = output_shape
    all_strides = [_get_var(strides, i) for i in range(n_blocks)]
    all_strides.append(final_strides)

    reshaped_w = w // 2 ** np.sum(np.array(all_strides) > 1)
    last_dense_size = reshaped_w * reshaped_w * conv_input_channels

    if not use_conv_as_final: n_blocks += 1
    
    if isinstance(noise_size, list):
        input_noise = [
            Input(shape = (size, ), name = 'input_{}'.format(i))
            for i, size in enumerate(noise_size)
        ]
        x = Concatenate()(input_noise)
    else:
        input_noise = Input(shape = (noise_size,), name = 'input_noise')
        x = input_noise
        
    if n_dense > 0:
        print("[WARNING]\t It is not recommanded to use Dense layer in generator model !")
        for i in range(n_dense):
            x = DenseBN(x, _get_var(dense_sizes, i), 
                        activation = _get_var(dense_activation, i), 
                        use_bias = _get_var(dense_bias, i), 
                        name = 'dense_{}'.format(i+1))
    
    x = DenseBN(
        x, last_dense_size, use_bias = final_dense_bias, bnorm = final_dense_bnorm,
        drop_rate = dense_drop_rate, activation = final_dense_activation,
        name = 'final_dense', ** kwargs
    )
    
    x = Reshape((reshaped_w, reshaped_w, conv_input_channels))(x)
    
    for i in range(n_blocks):
        final_layer = not use_conv_as_final and i == n_blocks -1
        config = {
            'filters'   : output_channels if final_layer else _get_var(filters, i), 
            'kernel_size' : final_kernel_size if final_layer else _get_var(kernel_size, i), 
            'strides'   : final_strides if final_layer else _get_var(strides, i), 
            'padding'   : 'same',
            'use_bias'  : False if final_layer else _get_var(use_bias, i), 
            'activation'    : final_activation if final_layer else _get_var(activation, i), 
            'bnorm'     : 'never' if final_layer else _get_var(bnorm, i), 
            'drop_rate' : 0. if final_layer else _get_var(drop_rate, i), 
            'name'  : 'generation_layer' if final_layer else 'upsampling_block_{}'.format(i+1),
            ** kwargs
        }
            
        x = Conv2DTransposeBN(x, ** config)
    
    if use_conv_as_final:
        if n_conv > 0:
            for i in range(n_conv):
                x = Conv2DBN(x,  
                             filters        = _get_var(conv_filters, i),
                             kernel_size    = _get_var(conv_kernel_size, i),
                             strides        = 1,
                             use_bias       = _get_var(conv_bias, i),
                             padding        = 'same',
                             dilation       = _get_var(conv_dilation, i),
                             activation     = _get_var(conv_activation, i),
                             bnorm          = _get_var(conv_bnorm, i),
                             drop_rate      = _get_var(conv_drop_rate, i),
                             pooling        = False,
                             name           = 'final_conv_{}'.format(i+1),
                             ** kwargs
                            )
    
        x = Conv2DTranspose(output_channels, kernel_size = final_kernel_size, 
                            strides = final_strides, padding = 'same', use_bias = False, 
                            activation = final_activation, name = 'generation_layer', ** kwargs)(x)
    
    outputs = x
    return tf.keras.Model(inputs = input_noise, outputs = outputs, name = name)


custom_functions    = {
    'simple_generator'  : simple_generator
}