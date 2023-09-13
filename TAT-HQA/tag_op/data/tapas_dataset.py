import re, copy
import string
import json
from tqdm import tqdm
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from typing import List, Dict, Tuple
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .file_utils import is_scatter_available
from tatqa_utils import  *
from .data_util import *
from .data_util import  _is_average, _is_change_ratio, _is_diff, _is_division, _is_sum, _is_times
from .derivation_split import infix_evaluator
from .mapping_split import split_mapping

# soft dependency
if is_scatter_available():
    from torch_scatter import scatter

def convert_start_end_tags(split_tags, paragraph_index):
    in_split_tags = split_tags.copy()
    split_tags = [0 for i in range(len(split_tags))]
    for i in range(len(in_split_tags)):
        if in_split_tags[i] == 1:
            current_index = paragraph_index[i]
            split_tags[i] = 1
            paragraph_index_ = paragraph_index[i:]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[j] == current_index:
                    split_tags[i+j] = 1
                else:
                    break
            break
    for i in range(1, len(in_split_tags)):
        if in_split_tags[-i] == 1:
            current_index = paragraph_index[-i]
            split_tags[-i] = 1
            paragraph_index_ = paragraph_index[:-i]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[-j] == current_index:
                    split_tags[-i-j] = 1
                else:
                    break
            break
    del in_split_tags
    return split_tags

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def sortFunc(elem):
    return elem[1]

def get_order_by_tf_idf(question, paragraphs):
    sorted_order = []
    corpus = [question]
    for order, text in paragraphs.items():
        corpus.append(text)
        sorted_order.append(order)
    tf_idf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()[1:]
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x:x[1])
    idx = [i[0] for i in sorted_similarities][::-1]
    return [sorted_order[index] for index in idx]

def get_answer_nums(table_answer_coordinates: List, paragraph_answer_coordinates: Dict):
    if table_answer_coordinates is not None:
        table_answer_num = len(table_answer_coordinates)
    else:
        table_answer_num = 0
    paragraph_answer_nums = 0
    if paragraph_answer_coordinates:
        for value in paragraph_answer_coordinates.values():
            paragraph_answer_nums += len(value)
    return table_answer_num, paragraph_answer_nums

def get_operands_index(label_ids, token_type_ids):
    row_ids = token_type_ids[:, :, 2]
    column_ids = token_type_ids[:, :, 1]
    max_num_rows = 64
    max_num_columns = 32
    row_index = IndexMap(
        indices=torch.min(row_ids, torch.as_tensor(max_num_rows - 1, device=row_ids.device)),
        num_segments=max_num_rows,
        batch_dims=1,
    )
    col_index = IndexMap(
        indices=torch.min(column_ids, torch.as_tensor(max_num_columns - 1, device=column_ids.device)),
        num_segments=max_num_columns,
        batch_dims=1,
    )
    cell_index = ProductIndexMap(row_index, col_index).indices
    first_operand_start = torch.argmax((label_ids!=0).int(), dim=1)[0]
    label_ids = label_ids[0, first_operand_start:]
    cell_index_first = cell_index[0, first_operand_start:]
    first_operand_end = torch.argmax(((cell_index_first-cell_index[0, first_operand_start])!=0).int())

    label_ids = label_ids[first_operand_end:]
    cell_index_first = cell_index_first[first_operand_end:]
    first_operand_end = first_operand_end+first_operand_start

    second_operand_start = torch.argmax((label_ids!=0).int())
    cell_index_second = cell_index_first[second_operand_start:]
    second_operand_end = torch.argmax(((cell_index_second-cell_index_first[second_operand_start])!=0).int())+second_operand_start
    second_operand_start+=first_operand_end
    second_operand_end+=first_operand_end
    return first_operand_start, first_operand_end, second_operand_start, second_operand_end

def get_tokens_from_ids(ids, tokenizer):
    tokens = []
    sub_tokens = []
    for id in ids:
        token = tokenizer._convert_id_to_token(id)
        if len(sub_tokens) == 0:
            sub_tokens.append(token)
        elif str(token).startswith("##"):
            sub_tokens.append(token[2:])
        elif len(sub_tokens) != 0:
            tokens.append("".join(sub_tokens))
            sub_tokens = [token]
    tokens.append("".join(sub_tokens))
    return "".join(tokens)

def get_number_mask(table):
    max_num_rows = 64
    max_num_columns = 32
    columns = table.columns.tolist()
    number_mask = np.zeros((1, max_num_columns*max_num_rows))
    number_value = np.ones((1, max_num_columns*max_num_rows)) * np.nan
    for index, row in table.iterrows():
        for col_index in columns:
            col_index = int(col_index)
            in_cell_index = (index+1)*max_num_columns+col_index+1
            table_content = row[col_index]
            number = to_number(table_content)
            if number is not None:
                number_mask[0, in_cell_index] = 1
                number_value[0, in_cell_index] = float(number)
    return number_mask, number_value

def tokenize_answer(answer):
    answer_tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(answer):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            answer_tokens.append(c)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                answer_tokens.append(c)
            else:
                answer_tokens[-1] += c
            prev_is_whitespace = False
    return answer_tokens

def string_tokenizer(string: str, tokenizer) -> List[int]:
    if not string:
        return []
    tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(string):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)

    split_tokens = []
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)

    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return ids


def table_tokenize(table, tokenizer, mapping, if_mapping):
    mapping_content = []
    table_cell_tokens = []
    table_ids = []
    table_tags = []
    table_if_tags = []
    table_cell_index = []
    table_cell_number_value = []
    table_mapping, if_table_mapping = False, False
    answer_coordinates, if_answer_coordinates = None, None

    if "table" in mapping and len(mapping["table"]) != 0:
        table_mapping = True
        answer_coordinates = mapping["table"]
    if "table" in if_mapping and len(if_mapping["table"]):
        if_table_mapping = True
        if_answer_coordinates = if_mapping["table"]
        
    current_cell_index = 1
    for i in range(len(table)):
        for j in range(len(table[i])):
            cell_ids = string_tokenizer(table[i][j], tokenizer)
            if not cell_ids:
                continue
            table_ids += cell_ids
            if is_number(table[i][j]):
                table_cell_number_value.append(to_number(table[i][j]))
            else:
                table_cell_number_value.append(np.nan)
            table_cell_tokens.append(table[i][j])
            if table_mapping:
                if [i, j] in answer_coordinates or (i, j) in answer_coordinates:
                    mapping_content.append(table[i][j])
                    table_tags += [1 for _ in range(len(cell_ids))]
                else:
                    table_tags += [0 for _ in range(len(cell_ids))]
            else:
                table_tags += [0 for _ in range(len(cell_ids))]
            if if_table_mapping:
                if [i, j] in if_answer_coordinates or (i, j) in if_answer_coordinates:
                    table_if_tags += [1 for _ in range(len(cell_ids))]
                else:
                    table_if_tags += [0 for _ in range(len(cell_ids))]
            else:
                table_if_tags += [0 for _ in range(len(cell_ids))]
            table_cell_index += [current_cell_index for _ in range(len(cell_ids))]
            current_cell_index += 1
    return table_cell_tokens, table_ids, table_tags, table_if_tags, table_cell_number_value, table_cell_index, mapping_content

def org_table_tokenize(table, tokenizer, mapping):
    table_tags = []
    table_mapping = False
    answer_coordinates = None

    if "table" in mapping and len(mapping["table"]) != 0:
        table_mapping = True
        answer_coordinates = mapping["table"]
    for i in range(len(table)):
        for j in range(len(table[i])):
            cell_ids = string_tokenizer(table[i][j], tokenizer)
            if not cell_ids:
                continue
            if table_mapping:
                if [i, j] in answer_coordinates:
                    table_tags += [1 for _ in range(len(cell_ids))]
                else:
                    table_tags += [0 for _ in range(len(cell_ids))]
            else:
                table_tags += [0 for _ in range(len(cell_ids))]
    return table_tags
def org_paragraph_tokenize(question, paragraphs, tokenizer, mapping):
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    split_tokens = []
    split_tags = []
    number_mask = []
    number_value = []
    tokens = []
    tags = []
    word_piece_mask = []
    paragraph_index = []

    paragraph_mapping = False
    paragraph_mapping_orders = []
    if "paragraph" in list(mapping.keys()) and len(mapping["paragraph"].keys()) != 0:
        paragraph_mapping = True
        paragraph_mapping_orders = list(mapping["paragraph"].keys())
    # apply tf-idf to calculate text-similarity
    sorted_order = get_order_by_tf_idf(question, paragraphs)
    for order in sorted_order:
        text = paragraphs[order]
        prev_is_whitespace = True
        answer_indexs = None
        if paragraph_mapping and str(order) in paragraph_mapping_orders:
            answer_indexs = mapping["paragraph"][str(order)]
        current_tags = [0 for i in range(len(text))]
        if answer_indexs is not None:
            for answer_index in answer_indexs:
                current_tags[answer_index[0]:answer_index[1]] = \
                    [1 for i in range(len(current_tags[answer_index[0]:answer_index[1]]))]

        start_index = 0
        wait_add = False
        for i, c in enumerate(text):
            if is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                prev_is_whitespace = True
            elif c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                tokens.append(c)
                tags.append(0)
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
        if wait_add:
            if 1 in current_tags[start_index:len(text)]:
                tags.append(1)
            else:
                tags.append(0)

    try:
        assert len(tokens) == len(tags)
    except AssertionError:
        print(len(tokens), len(tags))
        input()
    current_token_index = 1
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        for sub_token in sub_tokens:
            split_tags.append(tags[i])
        current_token_index+=1
    return split_tags


def question_if_part_tokenize(question, question_if_text, tokenizer, mapping, if_mapping):
    mapping_content = []
    split_tokens = []
    split_tags = []
    split_if_tags = []
    if_tags = []
    number_mask = []
    number_value = []
    tokens = []
    tags = []
    word_piece_mask = []
    paragraph_index = []
    is_if_part = []

    for text in [question]:
        prev_is_whitespace = True
        if_answer_indexs = None
        answer_indexs = None
        if "question" in if_mapping and if_mapping["question"]:
            if_answer_indexs = if_mapping["question"] # only one in question
        if "question" in mapping and mapping["question"]:
            answer_indexs = mapping["question"]
            # print('mapping has question', mapping["question"])
        current_tags = [0 for i in range(len(text))]
        current_if_tags = [0 for i in range(len(text))]

        if answer_indexs is not None:
            for answer_index in answer_indexs:
                mapping_content.append(text[answer_index[0]:answer_index[1]])
                current_tags[answer_index[0]:answer_index[1]] = \
                    [1 for i in range(len(current_tags[answer_index[0]:answer_index[1]]))]

        if if_answer_indexs is not None:
            for if_answer_index in if_answer_indexs:
                current_if_tags[if_answer_index[0]:if_answer_index[1]] = \
                    [1 for i in range(len(current_if_tags[if_answer_index[0]:if_answer_index[1]]))]

        start_index = 0
        wait_add = False
        
        question_if_part_index = [text.index(question_if_text),text.index(question_if_text) + len(question_if_text)]
        
        for i, c in enumerate(text):
            if is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    if_tags.append(1 if 1 in current_if_tags[start_index:i] else 0)
                    tags.append(1 if 1 in current_tags[start_index:i] else 0)
                    wait_add = False
                prev_is_whitespace = True
            elif c in ["-", "–", "~"]:
                if wait_add:
                    if_tags.append(1 if 1 in current_if_tags[start_index:i] else 0)
                    tags.append(1 if 1 in current_tags[start_index:i] else 0)
                    wait_add = False
                tokens.append(c)
                is_if_part.append(1 if i < question_if_part_index[1] and i >= question_if_part_index[0] else 0)
                if_tags.append(0)
                tags.append(0)
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    is_if_part.append(1 if i < question_if_part_index[1] and i >= question_if_part_index[0] else 0)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
        if wait_add:
            if_tags.append(1 if 1 in current_if_tags[start_index:len(text)] else 0)
            tags.append(1 if 1 in current_tags[start_index:len(text)] else 0)
    try:
        assert len(tokens) == len(if_tags)
        assert len(tokens) == len(is_if_part)
    except AssertionError:
        print(len(tokens), len(if_tags))
        input()
    #print("checking is_if_part")
    #print(tokens)
    #print(is_if_part)
    #print([tokens[i] for i in range(len(is_if_part)) if is_if_part[i]])
    current_token_index = 1
    question_if_part_position_id = []
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        number = to_number(token)
        if number is not None:
            number_value.append(float(number))
        else:
            number_value.append(np.nan)
        for sub_token in sub_tokens:
            split_if_tags.append(if_tags[i])
            split_tags.append(tags[i])
            question_if_part_position_id.append(is_if_part[i])
            split_tokens.append(sub_token)
            paragraph_index.append(current_token_index)
        current_token_index+=1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    question_ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return tokens, question_ids, split_tags, split_if_tags, word_piece_mask, number_mask, number_value, paragraph_index, question_if_part_position_id, mapping_content


def paragraph_tokenize(question, paragraphs, tokenizer, mapping, if_mapping):
    mapping_content = []
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    split_tokens = []
    split_tags = []
    split_if_tags = []
    number_mask = []
    number_value = []
    tokens = []
    tags = []
    if_tags = []
    word_piece_mask = []
    paragraph_index = []

    paragraph_mapping, paragraph_if_mapping = False, False
    paragraph_mapping_orders = []
    if "paragraph" in list(mapping.keys()) and len(mapping["paragraph"].keys()) != 0:
        paragraph_mapping = True
        paragraph_mapping_orders = list(mapping["paragraph"].keys())
    if "paragraph" in if_mapping and len(if_mapping["paragraph"]):
        paragraph_if_mapping = True
    # apply tf-idf to calculate text-similarity
    sorted_order = get_order_by_tf_idf(question, paragraphs)
    for order in sorted_order:
        text = paragraphs[order]
        prev_is_whitespace = True
        answer_indexs = None
        if_answer_indexs = None
        if paragraph_mapping and str(order) in paragraph_mapping_orders:
            answer_indexs = mapping["paragraph"][str(order)]
        if paragraph_if_mapping and str(order) in if_mapping["paragraph"]:
            if_answer_indexs = if_mapping["paragraph"][str(order)]
        current_tags = [0 for i in range(len(text))]
        current_if_tags = [0 for i in range(len(text))]
        if answer_indexs is not None:
            for answer_index in answer_indexs:
                mapping_content.append(text[answer_index[0]:answer_index[1]])
                current_tags[answer_index[0]:answer_index[1]] = \
                    [1 for i in range(len(current_tags[answer_index[0]:answer_index[1]]))]
        if if_answer_indexs is not None:
            for if_answer_index in if_answer_indexs:
                current_if_tags[if_answer_index[0]:if_answer_index[1]] = \
                    [1 for i in range(len(current_if_tags[if_answer_index[0]:if_answer_index[1]]))]
        start_index = 0
        wait_add = False
        for i, c in enumerate(text):
            if is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    if_tags.append(1 if 1 in current_if_tags[start_index:i] else 0)
                    wait_add = False
                prev_is_whitespace = True
            elif c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    if_tags.append(1 if 1 in current_if_tags[start_index:i] else 0)
                    wait_add = False
                tokens.append(c)
                tags.append(0)
                if_tags.append(0)
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
        if wait_add:
            if 1 in current_tags[start_index:len(text)]:
                tags.append(1)
            else:
                tags.append(0)
            if_tags.append(1 if 1 in current_if_tags[start_index:len(text)] else 0)
    try:
        assert len(tokens) == len(tags) == len(if_tags)
    except AssertionError:
        print(len(tokens), len(tags), len(if_tags))
        input()
    current_token_index = 1
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        number = to_number(token)
        if number is not None:
            number_value.append(float(number))
        else:
            number_value.append(np.nan)
        for sub_token in sub_tokens:
            split_tags.append(tags[i])
            split_if_tags.append(if_tags[i])
            split_tokens.append(sub_token)
            paragraph_index.append(current_token_index)
        current_token_index+=1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    paragraph_ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return tokens, paragraph_ids, split_tags, split_if_tags, word_piece_mask, number_mask, number_value, \
           paragraph_index, mapping_content



def _concat(question_and_if_ids,
            question_and_if_tags,
            question_and_if_if_tags,
            question_and_if_index,
            question_if_part_indicator,
            question_and_if_number_value,
            question_and_if_tokens,
            paragraph_ids,
            paragraph_tags,
            paragraph_if_tags,
            paragraph_index,
            paragraph_number_value,
            paragraph_tokens,
            table_ids,
            table_tags,
            table_if_tags,
            table_cell_index,
            table_cell_number_value,
            sep,
            question_length_limitation,
            passage_length_limitation,
            max_pieces,
            opt,
            num_ops,
            ari_tags):
    in_table_cell_index = table_cell_index.copy()
    in_paragraph_index = paragraph_index.copy()
    
    input_ids = torch.zeros([1, max_pieces])
    input_segments = torch.zeros_like(input_ids)
    paragraph_mask = torch.zeros_like(input_ids)
    paragraph_index = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    table_index = torch.zeros_like(input_ids)
    tags = torch.zeros_like(input_ids)
    if_tags = torch.zeros_like(input_ids)

    question_mask = torch.zeros_like(input_ids)
    
    question_if_part_attention_mask = torch.zeros_like(input_ids)

    opt_mask = torch.zeros_like(input_ids)
    ari_round_labels = torch.zeros([1,num_ops,input_ids.shape[1]])

    truncated_question = False
    if question_length_limitation is not None:
        if len(question_and_if_ids) > question_length_limitation:
            question_and_if_ids = question_and_if_ids[:question_length_limitation]
            question_if_part_indicator = question_if_part_indicator[:question_length_limitation]
            question_and_if_index = question_and_if_index[:question_length_limitation]
            truncated_question = True
    
    question_ids = [sep] + question_and_if_ids + [sep]
    question_if_part_indicator = [0] + question_if_part_indicator
    question_if_part_attention_mask[0, :len(question_if_part_indicator)] = torch.from_numpy(np.array(question_if_part_indicator))
    
    question_length = len(question_ids)
    question_mask[0,1:question_length-1] = 1
    table_length = len(table_ids)
    paragraph_length = len(paragraph_ids)
    if passage_length_limitation is not None:
        if len(table_ids) > passage_length_limitation:
            passage_ids = table_ids[:passage_length_limitation]
            table_length = passage_length_limitation
            paragraph_length = 0
        elif len(table_ids) + len(paragraph_ids) + 1> passage_length_limitation:
            table_length = len(table_ids)
            paragraph_length = passage_length_limitation - table_length - 1
            paragraph_ids = paragraph_ids[:paragraph_length]
            passage_ids = paragraph_ids + [sep] + table_ids
        else:
            passage_ids = paragraph_ids + [sep] + table_ids
            table_length = len(table_ids)
            paragraph_length = len(paragraph_ids)
    else:
        passage_ids = paragraph_ids + [sep] + table_ids

    passage_length = len(passage_ids)

    passage_ids = passage_ids + [sep]+ num_ops * [opt] + [sep]

    input_ids[0, :question_length] = torch.from_numpy(np.array(question_ids))
    input_ids[0, question_length:question_length + len(passage_ids)] = torch.from_numpy(np.array(passage_ids))
    qtp_attention_mask = input_ids != 0
    qtp_attention_mask = qtp_attention_mask.int()
    qtp_attention_mask = (1 - question_if_part_attention_mask) * qtp_attention_mask
    assert (qtp_attention_mask == -1).any() == False


    max_question_index = question_and_if_index[:question_length - 2][-1]
    if truncated_question == False:
        assert max_question_index == question_and_if_index[-1]
        assert len(question_and_if_number_value) == max_question_index


    paragraph_mask[0, 1: question_length - 1] = 1
    paragraph_mask[0, question_length:question_length + paragraph_length] = 1
    paragraph_index[0, 1:question_length - 1] = torch.from_numpy(np.array(question_and_if_index[:question_length - 2]))
    paragraph_index[0, question_length:question_length + paragraph_length] = \
        torch.from_numpy(np.array(in_paragraph_index[:paragraph_length])+ max_question_index)

    tags[0, question_length:question_length + paragraph_length] = \
        torch.from_numpy(np.array(paragraph_tags[:paragraph_length]))
    if_tags[0, question_length:question_length + paragraph_length] = \
        torch.from_numpy(np.array(paragraph_if_tags[:paragraph_length]))

    table_mask[0, question_length+ paragraph_length +1 :question_length + paragraph_length + 1 + table_length] = 1
    table_index[0, question_length+ paragraph_length + 1 :question_length + paragraph_length + 1 + table_length] = \
        torch.from_numpy(np.array(in_table_cell_index[:table_length]))
    
    tags[0, question_length+ paragraph_length + 1 :question_length + paragraph_length + 1 + table_length] = torch.from_numpy(np.array(table_tags[:table_length]))
    if_tags[0, question_length+ paragraph_length + 1 :question_length + paragraph_length + 1 + table_length] = torch.from_numpy(np.array(table_if_tags[:table_length]))
    

    opt_mask[0,passage_length +1  : passage_length + num_ops + 1 ] = 1
    
    
    tags[0, 1: question_length -1] = torch.from_numpy(np.array(question_and_if_tags[:question_length - 2]))
    if_tags[0, 1: question_length -1] = torch.from_numpy(np.array(question_and_if_if_tags[:question_length - 2]))
    
    
    # truncate these
    paragraph_number_value  = question_and_if_number_value[:max_question_index] + paragraph_number_value
    paragraph_tokens = question_and_if_tokens[:max_question_index] + paragraph_tokens

    if ari_tags != None:
      ari_table_tags = ari_tags["table"]
      ari_para_tags = ari_tags["para"]
      for i in range(num_ops):
         r_num_ops = len(ari_table_tags)
         if i >= r_num_ops:
             break
         if isinstance(ari_table_tags[i],dict):
             opd1_tags = np.array(ari_table_tags[i]["operand1"][:table_length])
             opd2_tags = np.array(ari_table_tags[i]["operand2"][:table_length])
             for j in np.where(opd2_tags == 1):
                 opd1_tags[j] = 1
             ari_round_labels[0,i,question_length+ paragraph_length + 1 :question_length + paragraph_length + 1 + table_length] = torch.from_numpy(opd1_tags)
             if paragraph_length > 1:
                p_opd1_tags = np.array(ari_para_tags[i]["operand1"][:paragraph_length])
                p_opd2_tags = np.array(ari_para_tags[i]["operand2"][:paragraph_length])
                for j in np.where(p_opd2_tags == 1):
                    p_opd1_tags[j] = 1
                ari_round_labels[0,i, question_length:question_length + paragraph_length] = torch.from_numpy(p_opd1_tags)
         else:
             ari_round_labels[0,i, question_length+ paragraph_length + 1 :question_length + paragraph_length + 1 + table_length] = torch.from_numpy(np.array(ari_table_tags[i][:table_length]))
             if paragraph_length > 1:
                ari_round_labels[0,i, question_length:question_length + paragraph_length] = torch.from_numpy(np.array(ari_para_tags[i][:paragraph_length]))
    
    return input_ids, qtp_attention_mask, question_if_part_attention_mask, paragraph_mask, paragraph_number_value, paragraph_index, paragraph_tokens, \
           table_mask, table_cell_number_value, table_index, tags, if_tags, input_segments,opt_mask,ari_round_labels,question_mask

def _test_concat(question_and_if_ids,
            question_and_if_index,
            question_if_part_indicator,
            question_and_if_number_value,
            question_and_if_tokens,
            paragraph_ids,
            paragraph_index,
            paragraph_number_value,
            paragraph_tokens,
            table_ids,
            table_cell_index,
            table_cell_number_value,
            sep,
            question_length_limitation,
            passage_length_limitation,
            max_pieces,
            opt,
            num_ops):
    in_table_cell_index = table_cell_index.copy()
    in_paragraph_index = paragraph_index.copy()
    
    input_ids = torch.zeros([1, max_pieces])
    input_segments = torch.zeros_like(input_ids)
    paragraph_mask = torch.zeros_like(input_ids)
    paragraph_index = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    table_index = torch.zeros_like(input_ids)
    question_if_part_attention_mask = torch.zeros_like(input_ids)
    opt_mask = torch.zeros_like(input_ids)

    question_mask = torch.zeros_like(input_ids)

    truncated_question = False
    if question_length_limitation is not None:
        if len(question_and_if_ids) > question_length_limitation:
            question_and_if_ids = question_and_if_ids[:question_length_limitation]
            question_if_part_indicator = question_if_part_indicator[:question_length_limitation]
            question_and_if_index = question_and_if_index[:question_length_limitation]
            truncated_question = True
    
    question_ids = [sep] + question_and_if_ids + [sep]
    question_if_part_indicator = [0] + question_if_part_indicator
    question_if_part_attention_mask[0, :len(question_if_part_indicator)] = torch.from_numpy(np.array(question_if_part_indicator))
    
    question_length = len(question_ids)
    question_mask[0,1:question_length-1] = 1
    table_length = len(table_ids)
    paragraph_length = len(paragraph_ids)
    if passage_length_limitation is not None:
        if len(table_ids) > passage_length_limitation:
            passage_ids = table_ids[:passage_length_limitation]
            table_length = passage_length_limitation
            paragraph_length = 0
        elif len(table_ids) + len(paragraph_ids) + 1> passage_length_limitation:
            table_length = len(table_ids)
            paragraph_length = passage_length_limitation - table_length - 1
            paragraph_ids = paragraph_ids[:paragraph_length]
            passage_ids = paragraph_ids + [sep] + table_ids
        else:
            passage_ids = paragraph_ids + [sep] + table_ids
            table_length = len(table_ids)
            paragraph_length = len(paragraph_ids)
    else:
        passage_ids = paragraph_ids + [sep] + table_ids

    passage_length = len(passage_ids)
    passage_ids = passage_ids + [sep]+ num_ops * [opt] + [sep]

    input_ids[0, :question_length] = torch.from_numpy(np.array(question_ids))
    input_ids[0, question_length:question_length + len(passage_ids)] = torch.from_numpy(np.array(passage_ids))
    qtp_attention_mask = input_ids != 0
    qtp_attention_mask = qtp_attention_mask.int()
    qtp_attention_mask = (1 - question_if_part_attention_mask) * qtp_attention_mask
    assert (qtp_attention_mask == -1).any() == False

    opt_mask[0,passage_length +1  : passage_length + num_ops + 1 ] = 1
    
    max_question_index = question_and_if_index[:question_length - 2][-1]
    if truncated_question == False:
        assert max_question_index == question_and_if_index[-1]
        assert len(question_and_if_number_value) == max_question_index

    paragraph_mask[0, 1: question_length - 1] = 1
    paragraph_mask[0, question_length:question_length + paragraph_length] = 1
    paragraph_index[0, 1:question_length - 1] = torch.from_numpy(np.array(question_and_if_index[:question_length - 2]))
    paragraph_index[0, question_length:question_length + paragraph_length] = \
        torch.from_numpy(np.array(in_paragraph_index[:paragraph_length])+ max_question_index)

    table_mask[0, question_length+ paragraph_length +1 :question_length + paragraph_length + 1 + table_length] = 1
    table_index[0, question_length+ paragraph_length + 1 :question_length + paragraph_length + 1 + table_length] = \
        torch.from_numpy(np.array(in_table_cell_index[:table_length]))
    
    # truncate these
    paragraph_number_value  = question_and_if_number_value[:max_question_index] + paragraph_number_value
    paragraph_tokens = question_and_if_tokens[:max_question_index] + paragraph_tokens
    
    return input_ids, qtp_attention_mask, question_if_part_attention_mask, paragraph_mask, paragraph_number_value, paragraph_index, paragraph_tokens, \
           table_mask, table_cell_number_value, table_index, input_segments,opt_mask,question_mask



"""
instance format:
input_ids: np.array[1, 512]. The input ids.
attention_mask: np.array[1, 512]. The attention_mask to denote whether a id is real or padded.
token_type_ids: np.array[1, 512, 3]. 
    The special tokens needed by tapas within following orders: segment_ids, column_ids, row_ids.
tags_label: np.array[1, 512]. The tag ground truth indicate whether this id is part of the answer.
paragraph_mask: np.array[1, 512]. 1 for ids belongs to paragraph, 0 for others
paragraph_word_piece_mask: np.array[1, 512]. 0 for sub-token, 1 for non-sub-token or the start of sub-token
paragraph_number_value: np.array[1, 512]. nan for no-numerical words and number for numerical words extract from current word. i.e.: $460 -> 460
table_number_value: np.array[1, max_num_columns*max_num_rows]. Definition is the same as above.
paragraph_number_mask: np.array[1, 512]. 0 for non numerical token and 1 for numerical token.
table_number_mask: np.array[1, max_num_columns*max_num_rows]. 0 for non numerical token and 1 for numerical token.
paragraph_index: np.array[1, 512], used to apply token-lv reduce-mean after getting sequence_output
number_order_label: int. The operator calculating order.
operator_label:  int. The operator ground truth.
scale_label: int. The scale ground truth.
answer: str. The answer used to calculate metrics.
"""

class TagTaTQAReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None, sep="<s>", op_mode:int=8,
                 ablation_mode:int=0,num_ops :int = 6):
        self.max_pieces = 512
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep = self.tokenizer._convert_token_to_id(sep)
        self.opt = self.tokenizer.encode("<OPT>")[1]
        self.skip_count = 0
        self.op_mode=op_mode

        self.num_ops = num_ops
        self.ari_ops = ARI_CLASSES_
                     
        if ablation_mode == 0:
            self.OPERATOR_CLASSES=OPERATOR_CLASSES_
        elif ablation_mode == 1:
            self.OPERATOR_CLASSES=get_op_1(op_mode)
        elif ablation_mode == 2:
            self.OPERATOR_CLASSES=get_op_2(op_mode)
        else:
            self.OPERATOR_CLASSES=get_op_3(op_mode)

    def _make_instance(self, input_ids, question_if_part_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, question_if_part_attention_mask,
                    paragraph_number_value, table_cell_number_value, question_if_part_number_value, paragraph_index, table_cell_index, question_if_part_index,
                    question_if_part_if_tags, number_order_label, tags, if_tags, operator_class, if_operator_class, scale_class,
                    paragraph_tokens, table_cell_tokens, answer_dict, question_id):
        return {
            "input_ids": np.array(input_ids),
            "question_if_part_ids": np.array(question_if_part_ids),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "question_if_part_attention_mask": np.array(question_if_part_attention_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "question_if_part_number_value": np.array(question_if_part_number_value),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_cell_index),
            "question_if_part_index": np.array(question_if_part_index),
            "question_if_part_if_tags": np.array(question_if_part_if_tags),
            "number_order_label": int(number_order_label),
            "tag_labels": np.array(tags),
            "if_tag_labels": np.array(if_tags),
            "operator_label": int(operator_class),
            "if_operator_label": int(if_operator_class),
            "scale_label": int(scale_class),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
        }



    def _to_instance(self, question_text: str, question_if_text: str, table: List[List[str]], paragraphs: List[Dict], answer_from: str,
                     answer_type: str, answer:str, counter_derivation: str, original_derivation: str, counter_facts:list,  original_answer_mapping: Dict,
                     if_mapping: Dict, if_operator:str, counter_answer_mapping:Dict, counter_scale: str, question_id: str, is_counter: int):
        
        question_and_if_tokens, question_and_if_ids, question_and_if_tags, question_and_if_if_tags, _,_, \
                question_and_if_number_value, question_and_if_index, question_if_part_indicator, _= \
            question_if_part_tokenize(question_text, question_if_text, self.tokenizer, original_answer_mapping, if_mapping)

        table_cell_tokens, table_ids, table_tags, table_if_tags, table_cell_number_value, table_cell_index, _ = table_tokenize(table, self.tokenizer, original_answer_mapping, if_mapping)
        paragraph_tokens, paragraph_ids, paragraph_tags, paragraph_if_tags, paragraph_word_piece_mask, paragraph_number_mask, \
                paragraph_number_value, paragraph_index, _= paragraph_tokenize(question_text, paragraphs, self.tokenizer, original_answer_mapping, if_mapping)
                         
        order_labels = np.zeros(self.num_ops)
        rounds_labels = -100
        if answer_type == "arithmetic":
            counter_operator_class = self.OPERATOR_CLASSES["ARITHMETIC"]
            num_facts = facts_to_nums(counter_facts)

            isavg = 0
            try:
               if _is_average(num_facts,answer):
                   ari_ops = [self.ari_ops['AVERAGE']]
                   ari_tags = {'table':[table_tags],'para':[paragraph_tags],'operation':[[0]*self.num_ops]}
                   isavg = 1
                   rounds_labels = 0
            except:
                isavg = 0
            if isavg == 0:
               dvt_split_suc = 0
               try:
                    ari_operations = infix_evaluator(original_derivation)
                    rounds_labels = len(ari_operations) - 1
                    dvt_split_suc = 1
                    if rounds_labels > self.num_ops - 1:
                        counter_operator_class = None
                        dvt_split_suc = 0
               except:
                   print("derivation split err")
                   counter_operator_class = None
               if dvt_split_suc == 1:
                   ari_ops = [self.ari_ops[i[0]] for i in ari_operations]
                   operands = [i[1:] for i in ari_operations]
                   ari_tags = {'table':[],'para':[],'operation':[]}
                   for i,opds in enumerate(operands): 
                       temp_mapping,operand_one_mapping,operand_two_mapping = split_mapping(opds,original_answer_mapping,table,paragraphs)
                       if temp_mapping == None:
                           counter_operator_class = None
                           break
                       else:
                           if ari_operations[i][0] in ['DIFF','DIVIDE']:
                              if "table" in operand_one_mapping and "table" in operand_two_mapping:
                                  if operand_one_mapping["table"][0][0] > operand_two_mapping["table"][0][0]:
                                      order_labels[i] = 1
                                  elif operand_one_mapping["table"][0][1] > operand_two_mapping["table"][0][1]:
                                      order_labels[i] = 1
                              elif "paragraph" in operand_one_mapping and "table" in operand_two_mapping:
                                  order_labels[i] = 0
                              elif "paragraph" in operand_one_mapping and "paragraph" in operand_two_mapping:
                                  opd1_pid = list(operand_one_mapping["paragraph"].keys())[0]
                                  opd2_pid = list(operand_two_mapping["paragraph"].keys())[0]
                                  if int(opd1_pid) > int(opd2_pid):
                                      order_labels[i] = 1
                                  elif operand_one_mapping["paragraph"][opd1_pid][0][0] > operand_two_mapping["paragraph"][opd2_pid][0][0]:
                                      order_labels[i] = 1

                              op1_table_tags = org_table_tokenize(table,self.tokenizer,operand_one_mapping)

                              op1_para_tags = org_paragraph_tokenize(question_text, paragraphs, self.tokenizer, operand_one_mapping)
                              op2_table_tags = org_table_tokenize(table,self.tokenizer,operand_two_mapping)
                              op2_para_tags = org_paragraph_tokenize(question_text, paragraphs, self.tokenizer, operand_two_mapping)
                              ari_tags['table'].append({"operand1":op1_table_tags,"operand2":op2_table_tags})
                              ari_tags['para'].append({"operand1":op1_para_tags,"operand2":op2_para_tags})
                              op1_tags = [0] * self.num_ops
                              op2_tags = [0] * self.num_ops
                              if 'operator' in operand_one_mapping:
                                  for i in operand_one_mapping['operator']:
                                      op1_tags[i] = 1
                              if 'operator' in operand_two_mapping:
                                  for i in operand_two_mapping['operator']:
                                      op2_tags[i] = 1
                              ari_tags['operation'].append({"operand1":op1_tags,"operand2":op2_tags})

                           else:
                              temp_table_tags = org_table_tokenize(table,self.tokenizer,temp_mapping)
                              temp_para_tags= org_paragraph_tokenize(question_text, paragraphs, self.tokenizer, temp_mapping)
                              ari_tags['table'].append(temp_table_tags)
                              ari_tags['para'].append(temp_para_tags)
                              temp_op_tags = [0] * self.num_ops
                              if 'operator' in temp_mapping:
                                  for i in temp_mapping['operator']:
                                      temp_op_tags[i] = 1
                              ari_tags['operation'].append(temp_op_tags)

        elif answer_type == "count":
            counter_operator_class = self.OPERATOR_CLASSES["COUNT"]
            ari_ops = None
            ari_tags = None
        else:
            counter_operator_class = get_operator_class(counter_derivation, answer_type, counter_facts, answer,
                                            counter_answer_mapping, counter_scale, self.OPERATOR_CLASSES)
            ari_ops = None
            ari_tags = None
        
        counter_scale_class = SCALE.index(counter_scale)
        if_operator_class = IF_OPERATOR_CLASSES_[if_operator]
        
        if counter_operator_class is None:
            self.skip_count += 1
            return None
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"
        table = pd.DataFrame(table, dtype=np.str_)
        column_relation = {}
        for column_name in table.columns.values.tolist():
            column_relation[column_name] = str(column_name)
        table.rename(columns=column_relation, inplace=True)
        
        
        concat_params = {"question_and_if_ids": question_and_if_ids, "question_and_if_tags": question_and_if_tags, "question_and_if_if_tags": question_and_if_if_tags,
                         "question_and_if_index": question_and_if_index, "question_if_part_indicator": question_if_part_indicator, "question_and_if_number_value": question_and_if_number_value, "question_and_if_tokens": question_and_if_tokens,
                         "paragraph_ids": paragraph_ids, "paragraph_tags": paragraph_tags, "paragraph_if_tags": paragraph_if_tags, "paragraph_index": paragraph_index,"paragraph_number_value": paragraph_number_value, "paragraph_tokens": paragraph_tokens,
                         "table_ids": table_ids, "table_tags": table_tags, "table_if_tags": table_if_tags, "table_cell_index": table_cell_index, "table_cell_number_value": table_cell_number_value,
                         "sep": self.sep,"question_length_limitation": self.question_length_limit, "passage_length_limitation": self.passage_length_limit, "max_pieces": self.max_pieces,
                         "opt":self.opt,"num_ops":self.num_ops,"ari_tags":ari_tags}
        
        input_ids, qtp_attention_mask, question_if_part_attention_mask, paragraph_mask, paragraph_number_value, paragraph_index, paragraph_tokens, \
        table_mask, table_cell_number_value, table_cell_index, tags, if_tags, input_segments, opt_mask,ari_round_labels,question_mask = _concat(**concat_params)

        opt_labels = torch.zeros(1,self.num_ops - 1 , self.num_ops-1)
        if answer_type == "arithmetic":
            ari_round_labels = torch.where(tags > 0 ,ari_round_labels,-100)
            if len(ari_ops) >= 2:
                for i in range(1 , len(ari_ops)):
                    opt_tags = ari_tags["operation"][i]
                    if isinstance(opt_tags , dict):
                        opd1_opt_tags = opt_tags["operand1"]
                        for j in range(self.num_ops-1):
                            if opd1_opt_tags[j] == 1:
                                opt_labels[0,j,i-1] = 1
                        opd2_opt_tags = opt_tags["operand2"]
                        for j in range(self.num_ops-1):
                            if opd2_opt_tags[j] == 1:
                                opt_labels[0,j,i-1] = 2
                    else:
                        for j in range(self.num_ops-1):
                            if opt_tags[j] == 1:
                                opt_labels[0,j,i-1] = 1

        
        answer_dict = {"answer_type": answer_type, "answer": answer, "scale": counter_scale, "answer_from": answer_from, "gold_if_op": if_operator}



        if ari_ops != None:
            ari_ops_padding = self.num_ops - len(ari_ops)
            if ari_ops_padding > 0:
               ari_ops += [0]*ari_ops_padding
            if ari_ops == [0] * self.num_ops:
                print("no ari ops")
                ari_ops = [-100] * self.num_ops
            else:
                get0 = False
                for i in range(self.num_ops):
                    if get0 == True:
                        ari_ops[i] = -100
                    if ari_ops[i] == 0:
                        if get0 == False:
                            get0 = True
        else:
            ari_ops = [-100] * self.num_ops

        if ari_ops == [-100] * self.num_ops:
            opt_labels[0,:,:] = -100
            ari_round_labels[0,:,:] = -100
            order_labels[:] = -100
            number_indexes = []
            ari_sel_labels = []
            if ari_ops[1] == 0:
                opt_labels[0,:,:] = -100
        else:
            print(ari_ops)
            for i in range(1,self.num_ops-1):
                for j in range(i):
                    opt_labels[0,i,j] = -100
                if ari_ops[i] == 0:
                    ari_round_labels[0,i:,:] = -100
                    opt_labels[0,i-1:,:] = -100
                    opt_labels[0,:,i-1:] = -100

            number_indexes = []
            cur_indexes = []
            cur = 0
            selected_indexes = torch.nonzero(tags[0]).squeeze(-1)
            
            if len(selected_indexes) == 0:
                print("no number")
                return None
            for sel in selected_indexes:
                sel = int(sel)
                if int(table_cell_index[0,sel]) != 0:
                    if int(table_cell_index[0,sel]) == cur or cur_indexes == []:
                        if cur_indexes == []:
                            cur = int(table_cell_index[0,sel])
                        cur_indexes.append(sel)
                    else:
                        cur = int(table_cell_index[0,sel])
                        number_indexes.append(cur_indexes)
                        cur_indexes = [sel]
                else:
                    if int(paragraph_index[0,sel]) == cur or cur_indexes == []:
                        if cur_indexes == []:
                            cur = int(paragraph_index[0,sel])
                        cur_indexes.append(sel)
                    else:
                        cur = int(paragraph_index[0,sel])
                        number_indexes.append(cur_indexes)
                        cur_indexes = [sel]
            number_indexes.append(cur_indexes)
            distinct_si = []
            for  i , ni in enumerate(number_indexes):
                distinct_si.append(ni[0])
                p = 10 - len(ni)
                if  p > 0:
                    number_indexes[i] += [0] * p
                else:
                    number_indexes[i] = number_indexes[i][:10]
                    print("long number")
                    print(ni)
                    print(table_cell_index[0,ni])
                    print(paragraph_index[0,ni])
                    if int(table_cell_index[0,ni[0]]) != 0 :
                        print(table_cell_number_value[int(table_cell_index[0,ni[0]]) - 1])
                    elif int(paragraph_index[0,ni[0]]) != 0:
                        print(paragraph_number_value[int(paragraph_index[0,ni[0]]) - 1])
                    else:
                        print("extract err")#if question_answer["uid"] in ignore_ids:

            ari_sel_labels = ari_round_labels[0,:,distinct_si].transpose(0,1)
            if ari_sel_labels.shape[0] != len(number_indexes):
                print(ari_sel_labels)
                print(number_indexes)
                exit(0)

            for i in range(self.num_ops):
                if ari_ops[i] in [2,4]:
                    count = 0
                    for sl in ari_sel_labels:
                        if sl[i] == 1:
                            count += 1
                    if count != 2:
                        order_labels[i] = -100
                else:
                    order_labels[i] = -100

        opt_id = torch.nonzero(opt_mask == 1)[0,1]

        ari_ops = ari_ops[:self.num_ops]
        make_instance = {
        "input_ids": np.array(input_ids), 
        "qtp_attention_mask":  np.array(qtp_attention_mask),
        "question_if_part_attention_mask": np.array(question_if_part_attention_mask),
        "token_type_ids": np.array(input_segments),
        "paragraph_mask":np.array(paragraph_mask), 
        "paragraph_number_value": np.array(paragraph_number_value), 
        "paragraph_index": np.array(paragraph_index), 
        "paragraph_tokens": paragraph_tokens,
        "table_mask": np.array(table_mask), 
        "table_cell_number_value": np.array(table_cell_number_value), 
        "table_cell_index": np.array(table_cell_index), 
        "table_cell_tokens": table_cell_tokens, 
        "tag_labels": np.array(tags), 
        "if_tag_labels": np.array(if_tags), 
        "operator_labels": int(counter_operator_class), 
        "rounds_labels" : int(rounds_labels),
        "if_operator_labels": int(if_operator_class), 
        "scale_labels": int(counter_scale_class),
        "answer_dict": answer_dict, 
        "question_id": question_id, 
        "is_counter_arithmetic": int(is_counter == 1 and answer_type == 'arithmetic'), 
        "is_original": 1 - is_counter,
        "ari_ops" : torch.LongTensor(ari_ops),
        "opt_mask" : opt_id,
        "order_labels" : torch.LongTensor(order_labels),
        "ari_labels" : torch.LongTensor(np.array(ari_sel_labels)),
        "selected_indexes" : np.array(number_indexes),
        "opt_labels": torch.LongTensor(np.array(opt_labels)),
        "question_mask" : np.array(question_mask)
        }
        
        return make_instance


    def _read(self, file_path: str):
        print("Reading train dataset file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        instances = []
        key_error_count = 0
        index_error_count = 0
        assert_error_count = 0
        reading_cnt = 1

        for one in tqdm(dataset):
            table = one['table']['table']
            paragraphs = one['paragraphs']
            questions = one['questions']
            #print('')
            #print("***** Reading *****", reading_cnt)
            reading_cnt += 1
            for question_answer in questions:
                try:
                    is_counter = question_answer["counterfactual"] # is counterfactual? 0 or 1
                    question = question_answer["question"]
                    answer_type = question_answer["answer_type"]
                    answer = question_answer["answer"]
                    answer_from = question_answer["answer_from"]
                    scale = question_answer["scale"]
                    if is_counter:
                        question_if_part = question_answer["question_if_part"]
                        if answer_type == 'arithmetic': # counterfactual original questions
                            original_answer_mapping = question_answer["original_answer_mapping"] # use original mapping to find original operands
                            counter_answer_mapping = question_answer["mapping"] # dummy, not used
                            facts = question_answer["counter_facts"] if "counter_facts" in question_answer else question_answer["facts"] # for deciding operator
                            counter_derivation = question_answer["derivation"] # for deciding operator
                            original_derivation = question_answer["original_derivation"] # not used
                            if_mapping = question_answer["if_mapping"] if "if_mapping" in question_answer else {}
                            if_operator = question_answer["if_op"] if "if_op" in question_answer else 'NONE'
                        else: # original arithmetic questions
                            original_answer_mapping = question_answer["mapping"] # only using the corresponding mapping and derivation
                            counter_answer_mapping = question_answer["mapping"]
                            facts = question_answer["facts"]
                            counter_derivation = question_answer["derivation"]
                            original_derivation = question_answer["derivation"]
                            if_operator = 'NONE'
                            if_mapping = {}
                    else:
                        question_if_part = ""
                        counter_derivation = question_answer["derivation"]
                        original_derivation = question_answer["derivation"]
                        if_mapping = {}
                        if_operator = 'NONE'
                        original_answer_mapping = question_answer["mapping"]
                        counter_answer_mapping = question_answer["mapping"]
                        facts = question_answer["facts"]
                    
                    instance = self._to_instance(question, question_if_part, table, paragraphs, answer_from,
                                    answer_type, answer, counter_derivation, original_derivation, facts, original_answer_mapping, if_mapping,
                                    if_operator, counter_answer_mapping, scale, question_answer["uid"], is_counter)
                    if instance is not None:
                        instances.append(instance)
                except RuntimeError as e :
                    pass
        print('total instance numbers', len(instances))
        return instances


class TagTaTQATestReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None, sep="<s>",
                 ablation_mode=0, op_mode=0, num_ops = 6 , mode ="dev"):
        self.max_pieces = 512
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep = self.tokenizer._convert_token_to_id(sep)
        self.opt = self.tokenizer.encode("<OPT>")[1]
        tokens = self.tokenizer._tokenize("Feb 2 Nov")
        self.skip_count = 0
        self.mode = mode
        self.num_ops = num_ops
        self.ablation_mode = ablation_mode
        if self.ablation_mode == 3:
            self.OPERATOR_CLASSES=get_op_3(op_mode)
        if ablation_mode == 0:
            self.OPERATOR_CLASSES=OPERATOR_CLASSES_
        self.op_count = {"Span-in-text":0, "Cell-in-table":0, "Spans":0, "Sum":0, "Count":0, "Arithmetic":0}
        self.scale_count = {"":0, "thousand":0, "million":0, "billion":0, "percent":0}
        

    def _make_instance(self, input_ids, question_if_part_input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, question_if_part_attention_mask,
                       paragraph_number_value, table_cell_number_value, question_if_part_number_value, paragraph_index, table_cell_index,
                       question_if_part_index, paragraph_tokens, table_cell_tokens, answer_dict, question_id,
                       paragraph_mapping_content, table_mapping_content):
        return {
            "input_ids": np.array(input_ids),
            "question_if_part_input_ids": np.array(question_if_part_input_ids),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "question_if_part_attention_mask": np.array(question_if_part_attention_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "question_if_part_number_value": np.array(question_if_part_number_value),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_cell_index),
            "question_if_part_index": np.array(question_if_part_index),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
        }

    def summerize_op(self, derivation, answer_type, facts, answer, answer_mapping, tv,pv):
        truth_numbers = []
        order_labels = [-100]*self.num_ops
        if answer_type == "span":
            if "table" in answer_mapping.keys():
                self.op_count["Cell-in-table"] += 1
                return ["Cell-in-table"] +["ignore"]* (self.num_ops-1),truth_numbers,order_labels
            elif "paragraph" in answer_mapping.keys():
                self.op_count["Span-in-text"] += 1
                return ["Span-in-text"] +["ignore"]* (self.num_ops-1) , truth_numbers,order_labels
        elif answer_type == "multi-span":
            self.op_count["Spans"] += 1
            return ["Spans"]  +["ignore"]* (self.num_ops-1),truth_numbers,order_labels
        elif answer_type == "count":
            self.op_count["Count"] += 1
            return ["Count"] +["ignore"] * (self.num_ops-1),truth_numbers,order_labels
        elif answer_type == "arithmetic":
            num_facts = facts_to_nums(facts)
            if not is_number(str(answer)):
                return "",truth_numbers
            else:
                self.op_count["Arithmetic"] += 1
                isavg = 0
                try:
                   if _is_average(num_facts,answer):
                       operator_classes = ["Average"]+["Stop"] +["ignore"]* (self.num_ops-2)
                       isavg = 1
                except:
                   isavg = 0
                if isavg == 0:
                    dvt_split_suc = 0
                    try:
                       ari_operations = infix_evaluator(derivation)
                       #print(len(ari_operations))
                       for ari in ari_operations:
                           for num in ari[1:]:
                               if not isinstance(num,str):
                                   if num not in truth_numbers:
                                       truth_numbers.append(num)
                       dvt_split_suc = 1
                       if len(ari_operations) > self.num_ops:
                           operator_classes = None
                           dvt_split_suc = 0
                    except:
                       print(derivation)
                       operator_classes = None
                    if dvt_split_suc == 1:
                        operator_classes = ["ignore"]*self.num_ops
                        for i,ari in enumerate(ari_operations):
                            if ari[0] == "SUM":
                                operator_classes[i] = "Sum"
                            if ari[0] == "DIFF":
                                operator_classes[i] = "Difference"
                                opd1 = -100
                                opd2 = -100
                                tl = len(tv)
                                pl = len(pv)
                                for o in range(tl+pl):
                                    if isinstance(ari[1],str) or isinstance(ari[2],str):
                                        break
                                    if o < pl:
                                        if pv[o] == ari[1]:
                                            opd1 = o
                                        if pv[o] == ari[2]:
                                            opd2 = o
                                    else:
                                        if tv[o-pl] == ari[1]:
                                            opd1 = o
                                        if tv[o-pl] == ari[2]:
                                            opd2 = o
                                if opd1 == -100 or opd2 == -100:
                                    print("order fail")
                                    order_labels[i] = -100
                                else:
                                    if opd1 <= opd2:
                                        order_labels[i] = 0
                                    else:
                                        order_labels[i] = 1
                            if ari[0] == "TIMES":
                                operator_classes[i] = "Multiplication"
                            if ari[0] == "DIVIDE":
                                operator_classes[i] = "Division"
                                opd1 = -100
                                opd2 = -100
                                tl = len(tv)
                                pl = len(pv)
                                for o in range(tl+pl):
                                    if isinstance(ari[1],str) or isinstance(ari[2],str):
                                        break
                                    if o < pl:
                                        if pv[o] == ari[1]:
                                            opd1 = o
                                        if pv[o] == ari[2]:
                                            opd2 = o
                                    else:
                                        if tv[o-pl] == ari[1]:
                                            opd1 = o
                                        if tv[o-pl] == ari[2]:
                                            opd2 = o
                                if opd1 == -100 or opd2 == -100:
                                    print("order fail")
                                    order_labels[i] = -100
                                else:
                                    if opd1 <= opd2:
                                        order_labels[i] = 0
                                    else:
                                        order_labels[i] = 1
                            if ari[0] == "AVERAGE":
                                operator_classes[i] = "Average"
                            j = i
                        #print(j)
                        if j < self.num_ops - 1:
                            operator_classes[j+1] = "Stop"
                    else:
                       operator_classes = None

            return operator_classes,truth_numbers,order_labels
                

    def _to_test_instance(self, question_text: str, question_if_text:str, table: List[List[str]], paragraphs: List[Dict], answer_from: str,
                          answer_type: str, answer:str, question_id:str, scale: str, original_answer_mapping,facts,original_derivation,counter_derivation):
        
        dummy_dict = {}
        
        table_cell_tokens, table_ids, _, _, table_cell_number_value, table_cell_index, _ = \
                            table_tokenize(table, self.tokenizer, dummy_dict, dummy_dict)

        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"
        table = pd.DataFrame(table, dtype=np.str_)
        column_relation = {}
        for column_name in table.columns.values.tolist():
            column_relation[column_name] = str(column_name)
        table.rename(columns=column_relation, inplace=True)

        paragraph_tokens, paragraph_ids, _, _, paragraph_word_piece_mask, paragraph_number_mask, \
                paragraph_number_value, paragraph_index, _= \
            paragraph_tokenize(question_text, paragraphs, self.tokenizer, dummy_dict, dummy_dict)



        question_and_if_tokens, question_and_if_ids, _, _, _,_, \
                question_and_if_number_value, question_and_if_index, question_if_part_indicator, _= \
            question_if_part_tokenize(question_text, question_if_text, self.tokenizer, dummy_dict, dummy_dict)

        concat_params = {"question_and_if_ids": question_and_if_ids,"question_and_if_index": question_and_if_index, "question_if_part_indicator": question_if_part_indicator, 
                         "question_and_if_number_value": question_and_if_number_value, "question_and_if_tokens": question_and_if_tokens,
                         "paragraph_ids": paragraph_ids, "paragraph_index": paragraph_index,"paragraph_number_value": paragraph_number_value, "paragraph_tokens": paragraph_tokens,
                         "table_ids": table_ids,  "table_cell_index": table_cell_index, "table_cell_number_value": table_cell_number_value,
                         "sep": self.sep, "question_length_limitation": self.question_length_limit, 
                         "passage_length_limitation": self.passage_length_limit, "max_pieces": self.max_pieces,"opt":self.opt,"num_ops":self.num_ops}

        input_ids, qtp_attention_mask, question_if_part_attention_mask, paragraph_mask, paragraph_number_value, paragraph_index, paragraph_tokens, \
        table_mask, table_cell_number_value, table_cell_index, input_segments,opt_mask,question_mask = _test_concat(**concat_params)

        opt_id = torch.nonzero(opt_mask == 1)[0,1]

        if self.mode == "dev" and original_derivation != None and facts != None and original_answer_mapping != None:
            gold_ops,truth_numbers,order_labels = self.summerize_op(original_derivation, answer_type, facts, answer, original_answer_mapping, table_cell_number_value,paragraph_number_value)
            if gold_ops is None:
                gold_ops = ["ignore"] *self.num_ops
        else:
            gold_ops= None
            order_labels = None
            truth_numbers = None
            
        answer_dict = {"answer_type": answer_type, "answer": answer, "answer_from": answer_from , "scale": scale,"gold_ops":gold_ops,"truth_numbers":truth_numbers,"order_labels":order_labels}

        self.scale_count[scale] += 1

        make_instance = {
            "input_ids": np.array(input_ids), 
            "qtp_attention_mask":  np.array(qtp_attention_mask),
            "question_if_part_attention_mask": np.array(question_if_part_attention_mask),
            "token_type_ids": np.array(input_segments), 
            "paragraph_mask":np.array(paragraph_mask), 
            "paragraph_number_value": np.array(paragraph_number_value), 
            "paragraph_index": np.array(paragraph_index), 
            "paragraph_tokens": paragraph_tokens,
            "table_mask": np.array(table_mask), 
            "table_cell_number_value": np.array(table_cell_number_value), 
            "table_cell_index": np.array(table_cell_index), 
            "table_cell_tokens": table_cell_tokens,
            "answer_dict": answer_dict, 
            "question_id": question_id,
            "opt_mask":opt_id,
            "question_mask" : np.array(question_mask)
            }
        
        return make_instance


    def _read(self, file_path: str):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        print("Reading the tatqa dataset")
        instances = []
        index_error_count = 0
        assert_error_count = 0
        reading_cnt = 1
        for one in tqdm(dataset):
            table = one['table']['table']
            paragraphs = one['paragraphs']
            questions = one['questions']
            reading_cnt += 1
            for question_answer in questions:
                try:
                    question = question_answer["question"]
                    if "question_if_part" in question_answer:
                        question_if_part = question_answer["question_if_part"]
                    else:
                        if question_answer["counterfactual"]:
                            question_if_part = get_question_if_part(question)
                        else:
                            question_if_part = ""
                    answer_type = question_answer["answer_type"] if "answer_type" in question_answer else ""
                    answer = question_answer["answer"] if "answer" in question_answer else ""
                    answer_from = question_answer["answer_from"] if "answer_from" in question_answer else ""
                    scale = question_answer["scale"] if "scale" in question_answer else ""

                    if "original_answer_mapping" in question_answer:
                        original_answer_mapping = question_answer["original_answer_mapping"]
                    elif "mapping" in question_answer:
                        original_answer_mapping = question_answer["mapping"]
                    else:
                        original_answer_mapping = None
                    if "counter_facts" in question_answer:
                        facts = question_answer["counter_facts"]
                    elif "facts" in question_answer:
                        facts = question_answer["facts"]
                    else:
                        facts = None
                    if "original_derivation" in question_answer:
                        original_derivation = question_answer["original_derivation"]
                    elif "derivation" in question_answer:
                        original_derivation = question_answer["derivation"]
                    else:
                        original_derivation = None
                    counter_derivation = question_answer["derivation"] if "derivation" in question_answer else None

                    
                    instance = self._to_test_instance(question, question_if_part, table, paragraphs, answer_from,
                                    answer_type, answer, question_answer["uid"], scale,original_answer_mapping,facts,original_derivation,counter_derivation)
                    if instance is not None:
                        instances.append(instance)
                except RuntimeError:
                    print(question_answer["uid"])
        print(self.scale_count)
        self.op_count = {"Span-in-text": 0, "Cell-in-table": 0, "Spans": 0, "Sum": 0, "Count": 0, "Arithmetic": 0}
        self.scale_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        print('total instances', len(instances))
        
        return instances

def get_question_if_part(question):
    start_if_word = ['If', 'Suppose','Given','When', 'Assuming']
    text = copy.deepcopy(question.lower())
    text = text[0].upper() + text[1:]
    question_first_word = text[:text.index(' ')]
    
    if_part = None
    w_part = None
    
    if question_first_word in start_if_word:
        if 'what' in text or 'in which' in text:
            ind = text.index('what') if 'what' in text else text.index('in which')
            if_part = question[:ind ]
            w_part = question[ind:]
        elif ', ' in text:
            ind = text.index(', ')
            if_part = question[:ind + 1]
            w_part = question[ind + 1:]
        elif '. ' in text:
            ind = ind = text.index('. ')
            if_part = question[:ind + 1]
            w_part = question[ind + 1:]
        else:
            pass
    else:
        if 'if' in text or 'when' in text or 'after' in text or 'given that' in text:
            if 'if ' in text:
                ind = text.index('if ')
            elif 'when ' in text:
                ind = text.index('when ')
            elif 'after ' in text:
                ind = text.index('after ')
            elif 'given that ' in text:
                ind = text.index('given that ')
            else:
                ind = len(text)
            if_part = question[ind:]
            w_part = question[:ind]
        else:
            pass
      
    if if_part is None or w_part is None:
        if_part = ''
        w_part = question

    assert if_part + w_part == question or w_part + if_part == question
    return if_part
        

# ### Beginning of everything related to segmented tensors ###

class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index
        Args:
            indices (:obj:`torch.LongTensor`, same shape as a `values` Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (:obj:`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (:obj:`int`, `optional`, defaults to 0):
                The number of batch dimensions. The first `batch_dims` dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.size()[: self.batch_dims]  # returns a torch.Size object

class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segments` * `inner_index.num_segments`
        Args:
            outer_index (:obj:`IndexMap`):
                IndexMap.
            inner_index (:obj:`IndexMap`):
                IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=(index.indices // self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
            .type(torch.float)
            .floor()
            .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )


def reduce_mean(values, index, name="segmented_reduce_mean"):
    """
    Averages a tensor over its segments.
    Outputs 0 for empty segments.
    This operations computes the mean over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a mean of
          vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the mean must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used
    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (:obj:`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "mean", name)

def flatten(index, name="segmented_flatten"):
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    `num_segments` * (k - 1). The result is a tensor with `num_segments` multiplied by the number of elements in the
    batch.
    Args:
        index (:obj:`IndexMap`):
            IndexMap to flatten.
        name (:obj:`str`, `optional`, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): The flattened IndexMap.
    """
    # first, get batch_size as scalar tensor
    batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    # next, create offset as 1-D tensor of length batch_size,
    # and multiply element-wise by num segments (to offset different elements in the batch) e.g. if batch size is 2: [0, 64]
    offset = torch.arange(start=0, end=batch_size, device=index.num_segments.device) * index.num_segments
    offset = offset.view(index.batch_shape())
    for _ in range(index.batch_dims, len(index.indices.size())):  # typically range(1,2)
        offset = offset.unsqueeze(-1)

    indices = offset + index.indices
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)

def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).
    Args:
        batch_shape (:obj:`torch.Size`):
            Batch shape
        num_segments (:obj:`int`):
            Number of segments
        name (:obj:`str`, `optional`, defaults to 'range_index_map'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = torch.as_tensor(
        batch_shape, dtype=torch.long
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    assert len(batch_shape.size()) == 1
    num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.size()) == 0

    indices = torch.arange(
        start=0, end=num_segments, device=num_segments.device
    )  # create a rank 1 vector with num_segments elements
    new_tensor = torch.cat(
        [torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)],
        dim=0,
    )
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(new_shape)

    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist())
    # equivalent (in Numpy:)
    # indices = torch.as_tensor(np.tile(indices.numpy(), multiples.tolist()))

    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])

def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.
    Args:
        values (:obj:`torch.Tensor`):
            Tensor with segment values.
        index (:obj:`IndexMap`):
            IndexMap.
        segment_reduce_fn (:obj:`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (:obj:`str`):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()) :]  # torch.Size object
    flattened_shape = torch.cat(
        [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    )
    # changed "view" by "reshape" in the following line
    flat_values = values.reshape(flattened_shape.tolist())

    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )

    # Unflatten the values.
    new_shape = torch.cat(
        [
            torch.as_tensor(index.batch_shape(), dtype=torch.long),
            torch.as_tensor([index.num_segments], dtype=torch.long),
            torch.as_tensor(vector_shape, dtype=torch.long),
        ],
        dim=0,
    )

    output_values = segment_means.view(new_shape.tolist())
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index
