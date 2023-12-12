import numpy as np
import re
import ast
import emoji

from sklearn.preprocessing import RobustScaler
IDEOGRAPHIC_SPACE = 0x3000

def is_asian(char):
    """Is the character Asian?"""
    return ord(char) > IDEOGRAPHIC_SPACE

def filter_jchars(c):
    """Filters Asian characters to spaces"""
    if is_asian(c):
        return ' '
    return c

def nonj_len(word):
    u"""Returns number of non-Asian words in {word}
    – 日本語AアジアンB -> 2
    – hello -> 1
    @param word: A word, possibly containing Asian characters
    """
    # Here are the steps:
    # 日spam本eggs
    # -> [' ', 's', 'p', 'a', 'm', ' ', 'e', 'g', 'g', 's']
    # -> ' spam eggs'
    # -> ['spam', 'eggs']
    # The length of which is 2!
    chars = [filter_jchars(c) for c in word]
    return len(''.join(chars).split())

def emoji_count(text):
    return len([i for i in text if i in emoji.EMOJI_DATA])

def get_wordcount(text):
    """Get the word/character count for text

    @param text: The text of the segment
    """

    characters = len(text)
    chars_no_spaces = sum([not x.isspace() for x in text])
    asian_chars =  sum([is_asian(x) for x in text])
    non_asian_words = nonj_len(text)
    emoji_chars = emoji_count(text)
    words = non_asian_words + asian_chars + emoji_chars

    return dict(characters=characters,
                chars_no_spaces=chars_no_spaces,
                asian_chars=asian_chars,
                non_asian_words=non_asian_words,
                emoji_chars = emoji_chars,
                words=words)

def dict2obj(dictionary):
    """Transform a dictionary into an object"""
    class Obj(object):
        def __init__(self, dictionary):
            self.__dict__.update(dictionary)
    return Obj(dictionary)

def get_wordcount_obj(text):
    """Get the wordcount as an object rather than a dictionary"""
    return dict2obj(get_wordcount(text))

def metadata(shorts_df):
    tags_cnt = []
    for _ in range(len(shorts_df['videoTags'])):
        if shorts_df['videoTags'][_] == 'none': tags_cnt.append(0)
        else: tags_cnt.append(len(ast.literal_eval(shorts_df['videoTags'][_])))
    m4_sort = shorts_df['videoDuration'].tolist()
    m4_sort_sec = []
    for i in range(len(m4_sort)):
        try: m4_sort_sec.append(int(m4_sort[i][2:4]))
        except: 
            try: m4_sort_sec.append(int(m4_sort[i][2:3]))
            except: continue
    title_length = []
    for _ in range(len(shorts_df)):
        a = shorts_df['videoTitle'][_]
        a = re.sub(r'[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+', "", a)
        b = get_wordcount_obj(a)
        # print(a)
        # print(b.words)
        title_length.append(b.words)
    descript_length = []
    for _ in range(len(shorts_df)):
        a = shorts_df['videoDescription'][_]
        try:
            # shorts_df['videoDescription'] = shorts_df['videoDescription'].replace({np.nan: None})
            a = re.sub(r'[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+', "", a)
        except TypeError:
            a = ''
        b = get_wordcount_obj(a)
        # print(a)
        # print(b.words)
        descript_length.append(b.words)
    scaler = RobustScaler() #StandardScaler()
    user_metadata = scaler.fit_transform(np.array(shorts_df['subscriberCount']).reshape(-1, 1))
    vid_duration = scaler.fit_transform(np.array(m4_sort_sec).reshape(-1, 1))
    vid_tags = scaler.fit_transform(np.array(tags_cnt).reshape(-1, 1))

    title_length = scaler.fit_transform(np.array(title_length).reshape(-1, 1))
    descript_length = scaler.fit_transform(np.array(descript_length).reshape(-1, 1))
    totalViewCount = scaler.fit_transform(np.array(shorts_df['totalViewCount']).reshape(-1, 1))
    totalVideoCount = scaler.fit_transform(np.array(shorts_df['totalVideoCount']).reshape(-1, 1))
    averageViewCount = scaler.fit_transform((np.array(shorts_df['totalViewCount']/shorts_df['totalVideoCount'])).reshape(-1, 1))
    
    ver1 = (vid_duration+vid_tags)*(title_length+descript_length) #[3.19035319] [0.00142437] !
    ver2 = (user_metadata/vid_tags) # [0.36745415] [0.71328627] -> nope
    ver3 = (user_metadata*vid_duration) # [2.44917069] [0.01433166]
    ver4 = (totalVideoCount/totalViewCount) # [16.3042378] [3.55510995e-59] !
    metadata_arr = np.column_stack([user_metadata, totalViewCount, totalVideoCount, averageViewCount, vid_duration, vid_tags, title_length, descript_length])
    print(metadata_arr.shape)
    return metadata_arr