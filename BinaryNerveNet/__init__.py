
# _*_ coding:utf-8 _*_
import tensorflow as tf
from numpy.random import RandomState
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# 在shape的一个维度上使用None可以方便使用不大的batch大小。在训练时需要把数据分成比较小的batch，但是
# 在测试时，可以一次性使用全部的数据。当数据集较小时这样比较方便测试，但数据集较大，将大量数据放入一个Batch可能会导致内存溢出
x=tf.placeholder(tf.float32,shape=(None,2),name ='x-input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

# 定义神经网络前向传播的过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# 定义损失函数和反响传播的算法
cross_entropy = -tf.reduce_mean(y_* tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy)
# 通过随机数声称一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X= rdm.rand(dataset_size,2)
# 定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格）
# 而其他为负样本(比如零件不合格)。和Tensorflow游乐场中的表示法不大一样的地方是，
# 在这里使用0表示负样本，1表示正样本。大部分解决分类问题的神经网络都会采用0，1表示
Y= [[int(x1+x2 <1)] for (x1,x2)in X]
# 创建一个会话来运行Tensorflow程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    # 初始化变量,并输出训练之前神经网络参数的值
    sess.run(init_op)
    print sess.run(w1)
    print sess.run(w2)
    # 设定训练的轮数
    STEPS =10000
    for i in range(STEPS):
        start=(i*batch_size)%dataset_size# 每次选取batch_size个样本进行训练
        end = min(start+batch_size,dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 1000==0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})# 每隔一段时间计算在所有数据上的交叉熵并输出
            print ("After %d training step(s),cross entropy on all data is %g "%(i,total_cross_entropy))
            # 通过这个结果我们可以发现随着训练的进行，交叉熵是逐渐变小的，交叉熵越小说明预测的结果和真是结果差距越小
            print sess.run(w1)
            print sess.run(w2)

