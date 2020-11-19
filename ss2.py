
from __future__ import print_function

import sys
import tensorflow.python.platform
import numpy as np
import random as random
from tensorflow.python.ops import rnn, rnn_cell_impl

import tensorflow.compat.v1 as tf

from roc import * 

tf.disable_v2_behavior()

print_prob = 0 
print_diff = 1 

check_min_cost = 0
min_cost = 0.0

#option
operation_mode = 0 # 0 for training and 1 for prediction
save_model = 1 #1 save 0 not save
save_summary = 0 #1 save 0 not save

batch_size = 128 #0 for batch 
learning_rate = 0.001
training_epochs = 20
display_step = 1
epsilon = 1e-3

bias_zero = 0

# Network Parameters
batch_norm = 0
keep_prob = 1.0
n_classes = 3

#arguments
train_file_name = sys.argv[1]+".train"
validate_file_name = sys.argv[1]+".validate"
test_file_name = sys.argv[1]+".test"

output_file_name = sys.argv[2]

beta = float(sys.argv[3]) # 0.00001 0.000005 0.000001 0.0000005
#beta2 = float(sys.argv[4]) 

batch_size=int(sys.argv[4]);
learning_rate = float(sys.argv[5]) 

keep_prob = float(sys.argv[6]) 

n_seq = int(sys.argv[7]) 
n_input = 20*n_seq

n_hidden = int(sys.argv[8])

model_no = -1 
if len(sys.argv) > 9:
    model_no = sys.argv[9]

if int(model_no) >= 0:
    operation_mode = 1

pred_flag = 0
if len(sys.argv) > 10:
    pred_flag = 1
    pred_file_name = sys.argv[10]

xv = tf.placeholder("float", [None,n_input])
xvr = tf.placeholder("float", [None,n_input])
yv = tf.placeholder("float", [None,n_classes])
lv = tf.placeholder(tf.int32, [None])

def read_data(filename):

    # Arrays to hold the labels and feature vectors.
    fvecs = []
    fvecrs = []
    labels = []
    lens = []

    # Iterate over the rows, splitting the label from the features. Convert labels
    # to integers and features to floats.

    count = 0
    with open(filename) as file:
        for line in file:
            row = line.strip().split(" ")

            if count%100000 == 0:
                print(count)

            count += 1

            #0 15 38 45 62 85 102 139 145 168 189 205
            labels.append(int(row[0]))

            #l = 1+(len(row)-1)//2
            #lens.append(l)

            lens.append(len(row)-1)

            fs = np.zeros(n_input)

            for x in row[1:]:
                fs[int(x)] = 1.0

            fvecs.append(fs)
            fvecrs.append(fs[::-1])

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)
    fvecrs_np = np.matrix(fvecrs).astype(np.float32)

    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)
    labels_onehot = (np.arange(n_classes) == labels_np[:, None]).astype(np.float32)

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np,fvecrs_np,labels_onehot,lens,labels


# Define weights tf.random_uniform
weights = {
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def output_at(outputs, ps):

    #batchxtimexstate
    b_size = tf.shape(outputs)[0]

    #outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.

    # Start indices for each sample
    index = tf.range(0, b_size) * n_seq + ps

    # Indexing
    output = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    return output


#seq: batch time hidden_state
#info batch protein_size
def  pred_model(seql, seqr, lens):

    batch_size = tf.size(seql)

    #batchxaax20
    seql = tf.reshape(seql,[-1,n_seq,20])
    seqr = tf.reshape(seqr,[-1,n_seq,20])

    #xx = tf.transpose(xx, [1, 0, 2])
    #xx = tf.reshape(xx, [-1, tok_dim])

    #cell_seq = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True, activation=tf.tanh)
    cell_seql = rnn_cell_impl.BasicLSTMCell(n_hidden, state_is_tuple=True)
    cell_seqr = rnn_cell_impl.BasicLSTMCell(n_hidden, state_is_tuple=True)

    with tf.variable_scope("rnn_f"):
        (output_seql,state_seql) = tf.nn.dynamic_rnn(cell_seql, seql, sequence_length=lens, time_major=False, dtype=np.float32)

    with tf.variable_scope("rnn_r"):
        (output_seqr,state_seqr) = tf.nn.dynamic_rnn(cell_seqr, seqr, sequence_length=lens, time_major=False, dtype=np.float32)

    outputl = output_at(output_seql, lens-1)
    outputr = output_at(output_seqr, lens-1)

    output = tf.concat([outputl, outputr],axis=1)

    pred = tf.matmul(output, weights['out']) + biases['out']

    return pred


with tf.variable_scope("rnn") as scope:
    do_pred = pred_model(xv, xvr, lv)
    do_class = tf.equal(tf.argmax(do_pred, 1), tf.argmax(yv, 1))
    do_accuracy = tf.reduce_mean(tf.cast(do_class, "float"))
    do_prob = tf.nn.softmax(do_pred)
    do_arg = tf.argmax(do_pred,1)

# Linear activation, using rnn inner loop last output

#cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=pred_train, labels=y, pos_weight))
#cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred, yy, pos_weight))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=do_pred, labels=yv))
cost += beta*tf.nn.l2_loss(weights["out"])


if save_summary == 1:
    tf.scalar_summary("cost", cost)

#with tf.device('/gpu:0'):
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#beta1=0.9, beta2=0.999 epsilon=1e-08

# Initializing the variables
init = tf.initialize_all_variables()

if operation_mode == 0:
    train_x, train_xr, train_y, train_l, _  = read_data(train_file_name)
    print("train data reading done")

    validate_x, validate_xr, validate_y, validate_l, _ = read_data(validate_file_name)
    print("validate data reading done")

test_x, test_xr, test_y, test_l, test_ls = read_data(test_file_name)
print("test data reading done")

if save_model == 1:
    saver = tf.train.Saver(max_to_keep=30)  # defaults to saving all variables - in this case w and b

# Launch the graph
with tf.Session() as sess:

    sess.run(init)

    if save_summary == 1:
        merged_summary = tf.merge_all_summaries()
        #summary_writer = tf.train.SummaryWriter('logs', sess.graph)
        summary_writer = tf.train.SummaryWriter('logs', graph=tf.get_default_graph())

    if operation_mode == 0:

        if int(model_no) >= 0:
            saver.restore(sess, output_file_name+'.model-'+model_no)
        
        prev_total_cost = 1000000 

        # Training cycle
        for epoch in range(training_epochs):
            
            total_cost = 0.0

            sample_no = len(train_y)

            batch_no = sample_no//batch_size
            if sample_no%batch_size != 0:
                batch_no = batch_no+1

            for b in range(0,batch_no):

                start = b*batch_size # random.randint(0,sample_no-batch_size)
                end = (b+1)*batch_size
                if end > sample_no:
                    end = sample_no
                    
                batch_x = train_x[start:end, :]
                batch_xr = train_xr[start:end, :]
                batch_y = train_y[start:end]
                batch_l = train_l[start:end]

                _,current_cost = sess.run([optimizer, cost], feed_dict={xv: batch_x, xvr: batch_xr, yv: batch_y, lv: batch_l})

                total_cost += current_cost

            #acc
            test_no = len(test_y)
            train_acc = do_accuracy.eval({xv: train_x[0:test_no], xvr: train_xr[0:test_no], yv: train_y[0:test_no], lv: train_l[0:test_no]});
            validate_acc = do_accuracy.eval({xv: validate_x, xvr: validate_xr, yv: validate_y, lv: validate_l});
            test_acc = do_accuracy.eval({xv: test_x, xvr: test_xr, yv: test_y, lv: test_l});

            # Display logs per epoch step
            #if epoch % display_step == 0:
            fo = open(output_file_name, "a")
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.5f}".format(total_cost), " validation=","{:.5f}".format(train_acc))
            fo.write("Epoch:")
            fo.write("%04d" % (epoch+1))
            fo.write(" cost=")
            fo.write("{:.5f}".format(total_cost))
            fo.write(" train=")
            fo.write("{:.5f}".format(train_acc))
            fo.write(" validation=")
            fo.write("{:.4f}".format(validate_acc))
            fo.write(" test=")
            fo.write("{:.4f}".format(test_acc))
            fo.write("\n")
            fo.close()

            if save_model == 1:
                save_path = saver.save(sess, output_file_name+'.model', global_step=epoch)

            if total_cost > 1.1*prev_total_cost:
                break

            prev_total_cost = total_cost

        print("Optimization Finished!")

    else:
        print(output_file_name+'.model-'+model_no)
        saver.restore(sess, "./"+output_file_name+'.model-'+model_no)

        #test     
        tfa = do_arg.eval({xv: test_x, xvr: test_xr, lv: test_l})
        tfy = do_pred.eval({xv: test_x, xvr: test_xr, yv: test_y, lv: test_l})
        tfp = do_prob.eval({xv: test_x, xvr: test_xr, yv: test_y, lv: test_l})
                                   
        print(len(test_ls))
        
        if pred_flag > 0:
            l0 = 0
            l1 = 0
            l2 = 0
            p0 = 0
            p1 = 0
            p2 = 0
            fo = open(pred_file_name, "a")
            for i in range(0, len(test_l)):
                if True:
                    if test_ls[i] == 0:
                        l0 = l0+1;
                        if tfp[i][0] > tfp[i][1] and tfp[i][0] > tfp[i][2]:
                            p0 = p0+1

                    if test_ls[i] == 1:
                        l1 = l1+1;
                        if tfp[i][1] > tfp[i][0] and tfp[i][1] > tfp[i][2]:
                            p1 = p1+1

                    if test_ls[i] == 2:
                        l2 = l2+1;
                        if tfp[i][2] > tfp[i][0] and tfp[i][2] > tfp[i][1]:
                            p2 = p2+1

                print("%d %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f" %(test_ls[i],tfy[i][0],tfy[i][1],tfy[i][2],tfp[i][0],tfp[i][1],tfp[i][2]), file=fo);
                #print(tfy[i][0],tfy[i][1],tfp[i][0],tfp[i][1],test_y[i][0],test_y[i][1], file=fo);
            fo.close()

        #for i in range(0, len(test_l)):
        #   print(tfy[i][0],tfy[i][1],tfp[i][0],tfp[i][1],test_y[i][0],test_y[i][1]);
              
        sample_no = len(test_l)

        #train_acc = do_accuracy.eval({xv: train_x[0:sample_no], xvr: train_xr[0:sample_no], yv: train_y[0:sample_no], lv: train_l[0:sample_no]});
        #validate_acc = do_accuracy.eval({xv: validate_x, xvr: validate_xr, yv: validate_y, lv: validate_l});
        test_acc = do_accuracy.eval({xv: test_x, xvr: test_xr, yv: test_y, lv: test_l});

        print("test = ",test_acc)

        if True:
            acc0 = (0.0+p0)/l0
            acc1 = (0.0+p1)/l1
            acc2 = (0.0+p2)/l2
            acc3 = (0.0+p0+p1+p2)/(l0+l1+l2)
            print(acc0,acc1,acc2,acc3)

        #calc_roc(tfp[:,1], tfa, test_y[:,1], 1)

