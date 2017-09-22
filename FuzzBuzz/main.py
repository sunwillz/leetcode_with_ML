# -*- coding: utf-8 -*-


from data_process import *
from sklearn.model_selection import train_test_split


import tensorflow as tf

# 测试集比例
testing_percentage = 0.3

# 训练参数
learning_rate = 0.03
num_steps = 1000
batch_size = 128
display_step = 100

# 模型参数
n_hidden = 16 # 单隐层网络节点数
num_input = num_size # 输入节点数
num_classes = 4 # 四分类问题输出节点数为4

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# 定义网络结构
def neural_net(input):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(input, weights['h1']), biases['b1']))
    output = tf.matmul(layer_1, weights['out']) + biases['out']
    return output


def main():
    dataSet, labels = generate_data()
    dataSet, labels = data_formation(labels)
    X_train, X_test, y_train, y_test = train_test_split(dataSet, labels, test_size=testing_percentage)
    print 'X_train size : ', X_train.shape
    print 'X_test size : ', X_test.shape
    print 'y_train : ', y_train.shape
    print 'y_test : ', y_test.shape

    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for step in range(1, num_steps + 1):
            for start in range(1,len(X_train),batch_size):
                # start = (step*batch_size) % data_size
                end = min(start+batch_size, data_size)
                sess.run(train_op, feed_dict={X: X_train[start:end], Y: y_train[start:end]})
            if step % display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_train[start:end], Y: y_train[start:end]})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")

        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: X_test, Y: y_test}))


if __name__== "__main__":
    main()