import re
import jieba
from typing import List

# 增加识别字符
jieba.re_han_default = re.compile(
    '([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-\xb7《》"\'‘’“”\(\)（）「」]+)', re.U)


def merge_char_with_previous(lst):
    """
    将列表中不能放在句子开头的元素，拼接到前一个元素后面。
    """
    new_lst = []

    for item in lst:
        chars_cannot_in_start = ['的', '了', '吗', '吧', '啊',
                                 '呀', '呢', '哦']
        if new_lst and (item in chars_cannot_in_start or new_lst[-1].isdigit()):
            new_lst[-1] += item
        else:
            new_lst.append(item)
    return new_lst


def combine_sentences(lst, max_length):
    """
    将一个列表lst分隔成多个子列表，确保每个子列表所有元素拼接后的长度不超过max_length，并且长度尽量均匀，并且子列表长度尽量少
    """

    def can_divide(lst, max_length, num_sublists):
        """
        检查是否可以将列表 lst 分为 num_sublists 个子列表，每个子列表长度不超过 max_length
        """
        current_length = 0
        count = 1
        for item in lst:
            item_length = len(item)
            if current_length + item_length > max_length:
                count += 1
                current_length = item_length
                if count > num_sublists:
                    return False
            else:
                current_length += item_length
        return True

    # 二分查找最优的子列表数量
    left, right = 1, len(lst)
    while left < right:
        mid = (left + right) // 2
        if can_divide(lst, max_length, mid):
            right = mid
        else:
            left = mid + 1

    # 使用最优子列表数量进行分割
    num_sublists = left
    total_length = sum(len(item) for item in lst)
    avg_length = total_length / num_sublists

    result = []
    current_sublist = []
    current_length = 0

    for item in lst:
        item_length = len(item)

        if current_length + item_length > max_length:
            result.append(current_sublist)
            current_sublist = [item]
            current_length = item_length
        else:
            current_sublist.append(item)
            current_length += item_length

        if current_length >= avg_length and len(result) < num_sublists - 1:
            result.append(current_sublist)
            current_sublist = []
            current_length = 0

    if current_sublist:
        result.append(current_sublist)

    combine_result = [''.join(r) for r in result]
    return combine_result


class WordSegmenter:
    def __init__(self):
        print("WordSegmenter initialized.")

    def add_word(self, word: str):
        """添加新词到分词器"""
        if word:
            jieba.add_word(word)
            print(f"Added word: {word}")
        else:
            print("No word to add.")

    def split_long_sentence(self, sentence: str, max_length) -> List[str]:
        segmented_words = self.segment_words(sentence)
        split_sentences = combine_sentences(segmented_words,
                                              max_length)
        return split_sentences


    def segment_words(self, text: str) -> List[str]:
        """对文本进行分词"""
        if not text:
            return []
        words = list(jieba.cut(text))
        words = merge_char_with_previous(words)
        return words
