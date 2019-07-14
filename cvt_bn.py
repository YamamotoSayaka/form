import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import tensorflow as tf
import pickle

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

filter_size = 17
in_channels = 26
out_channels = 28
inp = np.random.rand(1,100,26)


# tensorflow model
def weight_variable(shape, name='W'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)

def bias_variable(shape, name='b'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

def build_tf(incoming, scope=None, name='ResidualBlock_1d'):
    output = {}
    
    net = incoming
    with tf.variable_scope(scope, default_name=name, values=[incoming]) as scope:
        W1 = weight_variable([filter_size, in_channels, out_channels], name='W1')
        b1 = bias_variable([out_channels], name='b1')
        output_conv = tf.nn.conv1d(net, W1, stride=1, padding='SAME')+b1
        output_bn = tf.contrib.layers.batch_norm(output_conv, is_training=True)
    output['output_conv'] = output_conv
    output['output_bn'] = output_bn
    return output

input_tf_op  = tf.placeholder('float', shape=[None, None, 26], name='input_x1')
output_tf_op = build_tf(input_tf_op)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
output_tf = sess.run(output_tf_op, feed_dict={input_tf_op:inp})

vars = tf.trainable_variables()
weight_dict = {}
vars_vals = sess.run(vars)

for var, val in zip(vars, vars_vals):
    print(var.name, val.shape)
    weight_dict[var.name] = val



# pytorch model
def cc(inp):
    return nn.Parameter(torch.Tensor(inp))

def cconv(inp):
    inp = np.transpose(inp, [2,1,0])
    return nn.Parameter(torch.Tensor(inp))

class mm(nn.Module):
    def __init__(self):
        super(mm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=filter_size,
                                stride=1, padding=8)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def load_tf(self):
        self.conv1.weight = cconv(weight_dict['ResidualBlock_1d/Variable:0'])
        self.conv1.bias = cc(weight_dict['ResidualBlock_1d/Variable_1:0'])
        
        self.bn1.weight = cc(np.ones(28))
        self.bn1.bias = cc(weight_dict['ResidualBlock_1d/BatchNorm/beta:0'])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        return x1, x2

model_pytorch = mm()
model_pytorch.load_tf()
input_pytorch = torch.from_numpy(np.transpose(inp, [0, 2,1])).float()
output_pytorch_conv, output_pytorch_bn = model_pytorch(input_pytorch)



#########################################
# comparing the output
#########################################

output_pytorch_conv = output_pytorch_conv.data.cpu().numpy()
output_pytorch_bn = output_pytorch_bn.data.cpu().numpy()
output_tensorflow_conv = output_tf['output_conv'].transpose((0,2,1))
output_tensorflow_bn = output_tf['output_bn'].transpose((0,2,1))

diff_conv = np.abs(output_pytorch_conv - output_tensorflow_conv)
diff_bn = np.abs(output_pytorch_bn - output_tensorflow_bn)

print(np.max(diff_conv), np.max(diff_bn))
import pdb; pdb.set_trace()
