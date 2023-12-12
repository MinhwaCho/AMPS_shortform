import numpy as np
from imblearn.over_sampling import SMOTE

IMAGE_FEATURE_SIZE = 2048
TEXT_FEATURE_SIZE = 768
METADATA_SIZE = 8


def oversampling(text_shuffle_train,image_shuffle_train,y_vrank_shuffle_train,y_lrank_shuffle_train,y_shuffle_train):
    _text = text_shuffle_train
    _image = image_shuffle_train
    _y_vrank = y_vrank_shuffle_train
    _y_lrank = y_lrank_shuffle_train
    _y = y_shuffle_train

    image2 = _image #np.squeeze(_image, axis=1)
    rank_pd = _y_vrank
    rank_pd2 = _y_lrank
    feature = np.concatenate((_text,image2,rank_pd,rank_pd2),axis=1)

    sampling_strategy = 1
    smt = SMOTE(sampling_strategy=sampling_strategy)
    feature_over, y_over = smt.fit_resample(feature, _y)

    text_over = feature_over[:,:TEXT_FEATURE_SIZE]
    image_over = feature_over[:,TEXT_FEATURE_SIZE:TEXT_FEATURE_SIZE+IMAGE_FEATURE_SIZE]
    y_vrank_over = feature_over[:,TEXT_FEATURE_SIZE+IMAGE_FEATURE_SIZE:TEXT_FEATURE_SIZE+IMAGE_FEATURE_SIZE+TEXT_FEATURE_SIZE]
    y_lrank_over = feature_over[:,TEXT_FEATURE_SIZE+IMAGE_FEATURE_SIZE+TEXT_FEATURE_SIZE:]
    # print(text_over.shape, image_over.shape, y_vrank_over.shape,y_lrank_over.shape, y.shape)
    return text_over,image_over,y_vrank_over,y_lrank_over,y_over