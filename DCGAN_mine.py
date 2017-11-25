#coding:utf-8
"""
python 3.6
tensorflow 1.3
By LiWenDi
"""
#使用DCGAN网络架构,卷积/逆卷积成32x32x3的图像
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os.path
import cv2
import random
import time

toTrain = True #训练模式/输出模式
toContinueTrain = True #训练模式下继续上次的训练
toShuffle = True #是否打乱顺序
toShow = False #训练途中是否展示当前成果
howManyMake = 2 #每迭代多少次生成示例一次
howManySave = 2 #每迭代多少次保存模型一次
leaky_ReLU_alpha = 0.2 #leaky ReLU的负向保留值

new_train = True #是否允许重复训练直至loss达到要求
generator_first = True #优先满足生成器/辨别器的要求
generator_loss_demand = 1.0 #generator的loss要求
discriminator_loss_demand = 1.0 #discriminator的loss要求
max_re = 10 #最大允许重复训练次数
require_num = 5 #从这个迭代次数之后再开始重复训练
#生成图片
do_times = 10
generate_num = 10

#展示图片
show_batch_col = 4 #展示图片列数
show_batch_row = 3 #展示图片行数

output_path = "trained_model/"
sample_path = "samples_others/"
temp_path = "temp_samples/"
total_epoch = 150
batch_size = 25 #决定生成品的特异性，越小越具有特异性
learning_rate_generator = 0.00001
learning_rate_discriminator = 0.00001
n_width = 32
n_height = 32
n_input = n_width*n_height*3
n_noise = 128 #决定生成品共性，越大越离散
drop_prob = 0.75

#Discriminator权重矩阵限制范围
w_clip = 0.01

#--------------------读取数据-----------------------
input_data = "filedir"
image_dirs = os.listdir(input_data)
image_data = []
np.random.shuffle(image_dirs)
#print(image_dirs)
for image_dir in image_dirs:
    image_dir = os.path.join('%s/%s' % (input_data, image_dir))
    #print(image_dir)
    temp_pic = cv2.imread(image_dir,cv2.IMREAD_COLOR)
    b,g,r = cv2.split(temp_pic)  
    img = cv2.merge([r,g,b])
    image_data.append(cv2.resize(img, (n_height, n_width)))
    #image_data.append(cv2.imread(image_dir,cv2.IMREAD_GRAYSCALE))
dataset_num = len(image_data)
image_data = np.array(image_data, float)
image_data = image_data / 255.0

#print(image_data.shape)
#image_data = np.reshape(image_data, [len(image_data), n_height*n_width*3])
#----------------------------------------------------

if not toTrain:
    toContinueTrain = True
total_epoch = total_epoch*10 + 1

#-------------------------建立GAN网络----------------------------
#Descriminator网络输入图片形状
x = tf.placeholder(tf.float32,[None,n_height, n_width, 3])
#Generator网络输入的是噪声
z = tf.placeholder(tf.float32,[None,n_noise])
#keep_prob参数
keep_prob = tf.placeholder(tf.float32)
#Generator网络的权重和偏置

#构建Generator网络
def generator(noise_z):
    with tf.variable_scope('G', reuse=False):
        x1 = tf.layers.dense(noise_z, n_height*n_width*8)
        x1 = tf.reshape(x1, (-1, n_height//8, n_width//8, 512))
        x1 = tf.layers.batch_normalization(x1, training=True)
        x1 = tf.maximum(leaky_ReLU_alpha*x1, x1)
        #4x4x512

        x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=True)
        x2 = tf.maximum(leaky_ReLU_alpha*x2, x2)
        #8x8x256

        x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=True)
        x3 = tf.maximum(leaky_ReLU_alpha*x3, x3)
        #16x16x128

        logits = tf.layers.conv2d_transpose(x3, 3, 5, strides=2, padding='same')
        output = tf.tanh(logits)
        return output

#构建Discriminator网络
def discriminator(inputs, reuse=False):
    with tf.variable_scope('D', reuse=reuse):
        d1 = tf.layers.conv2d(inputs, 64, 5, strides=2, padding='same', name='d1')
        relu1 = tf.maximum(leaky_ReLU_alpha*d1, d1)
        #16x16x32

        d2 = tf.layers.conv2d(relu1, 128, 5,strides=2, padding='same', name='d2')
        bn2 = tf.layers.batch_normalization(d2, training=True)
        relu2 = tf.maximum(leaky_ReLU_alpha*bn2, bn2)
        #8x8x128

        d3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same', name='d3')
        bn3 = tf.layers.batch_normalization(d3, training=True)
        relu3 = tf.maximum(leaky_ReLU_alpha*bn3, bn3)
        #4x4x256

        flat = tf.reshape(relu3, (-1, n_height*n_width*4))
        logits = tf.layers.dense(flat, 1)
        output = tf.nn.sigmoid(logits)
        return logits
#----------------------------------------------------------------

#生成网络根据噪声生成一张图片
generator_output = generator(z)
#判别网络根据真实图片判别其真假概率
discriminator_real = discriminator(x)
#判别网络根据生成网络生成的图片片别其真假概率
discriminator_pred = discriminator(generator_output, True)

#生成网络loss
generator_loss_real =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_real, labels=tf.ones_like(tf.nn.sigmoid(discriminator_real))))
generator_loss_pred =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_pred, labels=tf.zeros_like(tf.nn.sigmoid(discriminator_pred))))
generator_loss = generator_loss_real + generator_loss_pred
#判别网络loss
discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_pred, labels=tf.ones_like(tf.nn.sigmoid(discriminator_pred))))

#生成网络loss
generator_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(discriminator_pred, 1e-6, 0.9999)))
#判别网络loss
discriminator_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(discriminator_real, 1e-6, 0.9999))+tf.log(1 - tf.clip_by_value(discriminator_pred, 1e-6, 0.9999)))


t_vars = tf.trainable_variables() #收集可训练的变量
d_vars = [var for var in t_vars if var.name.startswith('D')] #找出辨别器中的变量
g_vars = [var for var in t_vars if var.name.startswith('G')] #找出生成器中的变量

generator_train = tf.train.AdamOptimizer(learning_rate_generator).minimize(generator_loss,var_list=g_vars)
discriminator_train = tf.train.AdamOptimizer(learning_rate_discriminator).minimize(discriminator_loss,var_list=d_vars)
#截断clip
clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -w_clip, w_clip)) for var in d_vars]

other_vars = [var for var in tf.global_variables() if var not in t_vars] #找出应该被初始化的其他变量

saver = tf.train.Saver(var_list=t_vars)

with tf.Session() as sess:
    if os.path.exists(output_path+"model.ckpt.meta"):
        if toContinueTrain:
            saver.restore(sess, output_path+"model.ckpt")
            sess.run(tf.variables_initializer(other_vars))
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
    else:
        if toContinueTrain:
            print("找不到存储的模型文件！")
        init = tf.global_variables_initializer()
        sess.run(init)
    total_batch = int(dataset_num/batch_size)
    generator_c,discriminator_c = 0,0
    #开始交互模式
    #plt.ion()
    if toTrain:
        start_time = time.time()
        for epoch in range(total_epoch):
            if epoch % 10 == 1:
                start_time = time.time()
            if toShuffle:
                np.random.shuffle(image_data)
            for i in range(total_batch):
                batch_x = image_data[i * batch_size: (i + 1) * batch_size]
                noise = np.random.normal(size=(batch_size,n_noise))
                
                _,generator_c = sess.run([generator_train,generator_loss],feed_dict={x:batch_x,z:noise})
                _,discriminator_c = sess.run([discriminator_train,discriminator_loss],feed_dict={x:batch_x,z:noise})
                sess.run(clip_discriminator_var_op)
                if new_train:
                    if generator_first:
                        now_re = 0
                        while(discriminator_c > discriminator_loss_demand and now_re <= max_re and epoch >= require_num*10):
                            _,discriminator_c = sess.run([discriminator_train,discriminator_loss],feed_dict={x:batch_x,z:noise})
                            sess.run(clip_discriminator_var_op)
                            now_re = now_re + 1
                        now_re = 0
                        while(generator_c > generator_loss_demand and now_re <= max_re and epoch >= require_num*10):
                            _,generator_c = sess.run([generator_train,generator_loss],feed_dict={x:batch_x,z:noise})
                            now_re = now_re + 1
                    else:
                        now_re = 0
                        while(generator_c > generator_loss_demand and now_re <= max_re and epoch >= require_num*10):
                            _,generator_c = sess.run([generator_train,generator_loss],feed_dict={x:batch_x,z:noise})
                            now_re = now_re + 1
                        now_re = 0
                        while(discriminator_c > discriminator_loss_demand and now_re <= max_re and epoch >= require_num*10):
                            _,discriminator_c = sess.run([discriminator_train,discriminator_loss],feed_dict={x:batch_x,z:noise})
                            sess.run(clip_discriminator_var_op)
                            now_re = now_re + 1
            if epoch % 10 ==0:
                end_time = time.time()
                print('迭代次数: ',int(epoch/10 + 1),'--生成器_loss: %.4f' %generator_c,'--辨别器_loss: %.4f' %discriminator_c,'    耗时:{:.4f}(s)'.format((end_time-start_time)))
                print("----------------------------------------------------------------------------------")
                
                '''
                print("generator_output是：")
                print(sess.run(generator_output,feed_dict={z:noise}))
                print("discriminator_pred是：")
                print(sess.run(discriminator_pred,feed_dict={z:noise}))
                print("discriminator_real是：")
                print(sess.run(discriminator_real,feed_dict={x:batch_x}))
                '''
            if epoch % (10 * howManySave) == 0:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                saver.save(sess, output_path+"model.ckpt")
            #if epoch > 10*require_num:
                #discriminator_loss_demand = 0.4
                #generator_loss_demand = 0.8

            #图片显示
            if epoch % (10*howManyMake) == 0:
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                new_batch = show_batch_row * show_batch_col
                noise = np.random.normal(size=(new_batch,n_noise))
                #生成图像
                samples = sess.run(generator_output,feed_dict={z:noise})
                fig,a = plt.subplots(show_batch_row,show_batch_col)
                for i in range(show_batch_row):
                    for j in range(show_batch_col):
                        a[i][j].clear()
                        a[i][j].set_axis_off()
                        a[i][j].imshow(samples[show_batch_row*i + j])
                plt.draw()
                plt.savefig(temp_path+'temp_{0}.png'.format(epoch//10))
                if toShow:
                    plt.show()
                else:
                    plt.close(fig)
    else:
        for t in range(do_times):
            new_batch = generate_num
            noise = np.random.normal(size=(new_batch,n_noise))
            #生成图像
            samples = sess.run(generator_output,feed_dict={z:noise})
            fig,a = plt.subplots(1,generate_num,figsize=(2*generate_num,2))
            for i in range(new_batch):
                a[i].set_axis_off()
                a[i].imshow(samples[i])
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            plt.savefig(sample_path+'output_{0}.png'.format(t))
            plt.close(fig)
            print("生成了第{0}幅图片".format(t+1))
