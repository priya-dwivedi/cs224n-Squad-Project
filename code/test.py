# import numpy as np
#
# a = np.random.randint(10, size=10)
#
# print(a)
#
# # b = a[6:10]
# #
# # print(b)
# #
# # pos = np.argmax(b)
# #
# # print(pos)
#
# d = dict(enumerate(a,2))
#
# print(d)


def create_char_dicts(CHAR_PAD_ID=0, CHAR_UNK_ID=1, _CHAR_PAD='*', _CHAR_UNK='$'):
    unique_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                    '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '[', ']', '^', 'a', 'b', 'c', 'd',
                    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                    'y', 'z',
                    '~', ]  # based on analysis in jupyter notebook

    num_chars = len(unique_chars)

    idx2char = dict(enumerate(unique_chars, 2))  ##reserve first 2 spots
    idx2char[CHAR_PAD_ID] = _CHAR_PAD
    idx2char[CHAR_UNK_ID] = _CHAR_UNK

    ##Create reverse char2idx
    char2idx = {v: k for k, v in idx2char.iteritems()}
    return char2idx, idx2char


char2idx, idx2char = create_char_dicts()

print(char2idx)
print(len(char2idx))