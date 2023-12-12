import keras
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Reshape, Dense, Dropout, LSTM, Concatenate, Flatten, Activation, MaxPool2D, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import optimizers
from keras_self_attention import SeqSelfAttention

from utils import selfDef
from utils.selfDef import myLossFunc, Attention, coAttention_para, zero_padding, tagOffSet

IMAGE_FEATURE_SIZE = 2048
TEXT_FEATURE_SIZE = 768
METADATA_SIZE = 8

NUM_CLASSES = 1
LEARN_RATE = 0.0001 #0.01 * (BATCH_SIZE / 128) #try increasing batch size!

def AMPS(num_classes=NUM_CLASSES, learn_rate=LEARN_RATE):
    inputs_img = Input(shape=(IMAGE_FEATURE_SIZE,), name='i_input')
    inputs_text = Input(shape=(TEXT_FEATURE_SIZE,), name='t_input')
    inputs_like_perday = Input(shape=(TEXT_FEATURE_SIZE,), name='a_input')
    inputs_sub_num = Input(shape=(METADATA_SIZE,), name='b_input')

    wt_init = keras.initializers.RandomNormal(mean=0, stddev=0.01)
    bias_init = keras.initializers.Constant(value=0.5)

    def dense_layer(**args):
        return keras.layers.Dense(**args, 
            kernel_initializer=wt_init, 
            bias_initializer=bias_init)
    
    main_branch_t = Reshape((1,TEXT_FEATURE_SIZE))(inputs_text)
    main_branch_t = Bidirectional(LSTM(16))(main_branch_t) # LSTM (batch_size, timesteps, features)
    main_branch_t = Reshape((1,32))(main_branch_t)
    main_branch_t = SeqSelfAttention(attention_activation='sigmoid')(main_branch_t)
    # main_branch_t = Dense(32,activation='sigmoid')(main_branch_t)
    # main_branch_t = LSTM(32)(main_branch_t)
    main_branch_t = Dropout(0.4)(main_branch_t)
    main_branch_t = Reshape((1,32))(main_branch_t)

    main_branch_i = Reshape((1,IMAGE_FEATURE_SIZE))(inputs_img)
    main_branch_i = Dense(256,activation='sigmoid')(main_branch_i)
    main_branch_i = Dense(128,activation='sigmoid')(main_branch_i)
    main_branch_i = Dropout(0.2)(main_branch_i)
    main_branch_i = Reshape((1,128,))(main_branch_i)
    main_branch_i = Bidirectional(LSTM(16))(main_branch_i)
    main_branch_i = Reshape((1,32,))(main_branch_i)
    main_branch_i = SeqSelfAttention(attention_activation='sigmoid')(main_branch_i)
    # main_branch_i = Dense(32,activation='sigmoid')(inputs_text)
    # main_branch_i = LSTM(32)(main_branch_i)
    main_branch_i = Dropout(0.2)(main_branch_i)
    main_branch_i = Reshape((1,32,))(main_branch_i)

    dim_k = 32 #128
    
    like_per = Reshape((1,TEXT_FEATURE_SIZE))(inputs_like_perday)
    like_per = Bidirectional(LSTM(16))(like_per)
    like_per = Reshape((1,32))(like_per)
    like_per = SeqSelfAttention(attention_activation='sigmoid')(like_per)
    # like_per = LSTM(32)(inputs_like_perday)
    # like_per = Dense(32,activation='sigmoid')(inputs_like_perday)
    like_per = Dropout(0.4)(like_per)
    like_per = Reshape((1,32))(like_per)
    
    main_branch_1 = coAttention_para(dim_k=dim_k)([main_branch_t, main_branch_i])
    main_branch_2 = coAttention_para(dim_k=dim_k)([main_branch_t, like_per])
    main_branch_3 = coAttention_para(dim_k=dim_k)([main_branch_i, like_per])

    main_branch_t = Flatten()(main_branch_t)
    main_branch_i = Flatten()(main_branch_i)
    like_per = Flatten()(like_per)
    sub_num = Flatten()(inputs_sub_num)

    main_branch = Concatenate()([main_branch_1,main_branch_2,main_branch_3,main_branch_t,main_branch_i,like_per])

    
    task_1_branch = Dense(32, activation='sigmoid')(main_branch)
    # task_1_branch = Dropout(0.2)(task_1_branch)
    task_1_branch = Dense(8, activation='sigmoid')(task_1_branch)
    task_1_branch = Dropout(0.2)(task_1_branch)
    task_1_branch = Concatenate()([task_1_branch,sub_num])
    # task_1_branch = Dense(4, activation='sigmoid')(main_branch)
    # task_1_branch = Dropout(0.2)(task_1_branch)

    task_1_branch = dense_layer(units=num_classes, activation='sigmoid', name='T1')(task_1_branch)
    
    # # task_2_branch = LSTM(128)(main_branch)
    # task_2_branch = Dropout(0.5)(main_branch_t)
    # task_2_branch = Flatten()(task_2_branch)
    # task_2_branch = Dense(64, activation='sigmoid')(task_2_branch)
    # task_2_branch = Dropout(0.2)(task_2_branch)
    task_2_branch = dense_layer(units=num_classes, name='T2')(main_branch)

    model = Model(inputs=[inputs_img, inputs_text, inputs_like_perday,inputs_sub_num],
                      outputs=[task_1_branch,task_2_branch])
    # model.summary()
    dj_loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2)
    opt_adam = tf.keras.optimizers.Adam(learning_rate=learn_rate,epsilon=1e-06)
    
    model.compile(loss ={'T1': dj_loss,
                         'T2': 'mse'},\
                  optimizer = opt_adam,\
                  metrics={'T1':['accuracy'],
                           'T2':['mse','mae']})
    return model