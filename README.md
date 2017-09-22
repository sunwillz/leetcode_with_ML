###  <center>用机器学习解决LeetCode问题</center>

#### 导语

机器学习是一种使以计算机为代表的电子设备具备一定智能的技术，是人工智能的重要技术组成之一。特别是近年来以深度学习为代表的机器学习迅猛发展，使得人们的生活逐渐发生质的变化，机器学习从来没有像今天这样受到人们的热切关注。其实机器学习并不是高端到不可亵玩的东西，它只是利用计算机强大的计算能力来解决不能直接求解的数学问题，也就是说让计算机在某一个假设空间内寻找一个映射函数，将输入映射为我们需要的结果。既然我们给计算机规定了任务，为了让他自动的完成这个任务，我们当然需要告诉他该怎么做，虽然从表面上看不出来我们对计算机做了什么，但他已经从大量的数据中学得了有用的经验，这样足以使得计算机从表面上看有了一定的智能。

传统的数据结构与算法可以用编程语言解决，这种解决方法的核心在于时间复杂度和空间复杂度上的优化和提升，给定输入，计算机按照算法的指示一步步运算，输出对应的结果，这是一种显式求解的过程，并具有很强的解释性。那能不能通过机器学习解决传统的算法问题呢？我们试试看。

下面的内容包括:

+ 待解决的算法问题是什么
+ 用传统的机器学习算法SVM、logistics regression、gbdt、Xgboost求解
+ 用深度卷积神经网络求解
+ 比较上述算法的效果


#### LeetCode问题描述

机器学习虽然强大，但到目前为止，还不能解决所有的问题，因此考虑到可行性，选取了一道具有代表性的算法题，尝试利用机器学习求解。

[leetcode[412]:Fuzz Buzz](https://leetcode.com/problems/fizz-buzz/description/) 是LeetCode中难度为easy的题目，题目本身并没有什么难度，关键是如何将它转化为机器学习任务，其问题描述如下：
> 给定一个正整数n，按规则输出从1至n之间的数对应的结果。规则为：若数是3的倍数，输出“Fizz”，若数是5的倍数，则输出“Buzz”,若既是3的倍数也是5的倍数，则对应的输出为“FizzBuzz”,否则输出原来的数。
> 
> 举例：   
> n = 15,  
> Return:  
> [   
>   "1",  
>   "2",  
>   "Fizz",  
>   "4",  
>   "Buzz",  
>   "Fizz",  
>   "7",  
>   "8",  
>   "Fizz",  
>   "Buzz",  
>   "11",  
>   "Fizz",  
>   "13",  
>   "14",  
>   "FizzBuzz"  
> ] 

该题若用Python，解法如下：
   
    class Solution(object):
	    def fizzBuzz(self, n):
	        """
	        :type n: int
	        :rtype: List[str]
	        """
	        res = []
	        for i in range(1,n+1):
	            if i%3==0 and i%5==0:
	                res.append('FizzBuzz')
	            elif i%3==0:
	                res.append('Fizz')
	            elif i%5==0:
	                res.append('Buzz')
	            else:
	                res.append(str(i))
	        return res
	        
	        
现在我们将算法题转化为机器学习问题，计算机能否准确识别题目本来的含义，从而正确的给出我们想要得到结果呢？

#### 问题转化

按照题目要求，输出的结果共有4类：Fuzz、Buzz、FuzzBuzz、other，other表示数字本身，机器学习的目的就是自动学得题目的要求，这就是一个四分类的监督学习问题。  

我将构造一个极简单的单隐层的神经网络，输入数据量为4096条，对应的标签也为4096。由于是四分类问题，故标签维度为4，采用one-hot表示法。输入数据表示一个正整数，此处有多种表示方法，比如位表示法，binary表示法等，为了使得模型尽量简单，输入数据维度尽量低，因此采用binary表示法，即将输入的正整数表示成二进制，我构造了4096条数据，对应2的12次方，因此输入维度为12.

接下来就是构造神经网络了。我采用了TensorFlow框架，对于上述问题而言，单隐层的网络已足够解决问题，激活函数采用非线性化relu,可以加快模型的训练速度。

划分数据集，训练集和测试集的比例是7：3，要想获得好的模型效果，需要不断调整模型参数，因此需要多次试验。由于过程比较简单，下面直接上代码。

##### 构造数据集

	
    
    # -*- coding: utf-8 -*-

	import numpy as np
	
	# 数据总量,2**data_size
	num_size = 12
	data_size = 2**num_size
	
	
	def fizzBuzz(n):
	    """
	    :type n: int
	    :rtype: List[str]
	    """
	    res = []
	    for i in range(1, n + 1):
	        if i % 3 == 0 and i % 5 == 0:
	            res.append('FizzBuzz')
	        elif i % 3 == 0:
	            res.append('Fizz')
	        elif i % 5 == 0:
	            res.append('Buzz')
	        else:
	            res.append(str(i))
	    return res
	
	def generate_data():
	    dataSet = np.arange(1, data_size+1)
	    labels = fizzBuzz(data_size)
	
	    return np.array(dataSet), np.array(labels)
	
	
	def label_to_categorical(data):
	    labels = []
	    for i in range(len(data)):
	        if data[i] == 'Fizz':
	            labels.append([1, 0, 0, 0])
	        elif data[i] == 'Buzz':
	            labels.append([0, 1, 0, 0])
	        elif data[i] == 'FizzBuzz':
	            labels.append([0, 0, 1, 0])
	        else:
	            labels.append([0, 0, 0, 1])
	
	    return np.array(labels)
	
	
	def binary_encode(i, num_digits):
	    return np.array([i >> d & 1 for d in range(num_digits)])
	
	
	def data_formation(labels):
	    labels = label_to_categorical(labels)
	    data_df = np.array([binary_encode(i, num_size) for i in range(2 ** num_size)])
	    return data_df, labels


##### 构造模型训练并测试性能

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
	    # 划分数据集
	    X_train, X_test, y_train, y_test = train_test_split(dataSet, labels, test_size=testing_percentage)
	    print X_train.shape, X_test.shape, y_train.shape, y_test.shape
	
	    logits = neural_net(X)
	    prediction = tf.nn.softmax(logits)
		 # 损失函数
	    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	    train_op = optimizer.minimize(loss_op)
		 # 模型准确率
	    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		 # 初始化
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


训练过程和效果如下图：
<center>![](1.png)</center>

#### 结论

从上图可以看出在训练集上损失降到了0，准确率达到了100%，模型完全拟合了训练集，因为任务的函数空间很小，而含单隐层的神经网络理论上可以表示任意的函数，只要训练轮数足够，可以完全拟合训练集。测试集上的准确率也很高，达到了97.8%，可以说模型充分学习了我们给他的任务，也就是说给定任意一个正整数，模型给出的预测97.8%的概率和标准结果相等，可以说模型基本完成了学习任务。
