# Tokenize all texts and align the labels with them.
def preprocess_wordlevel(
    examples, src_lang, tgt_lang, tokenizer, padding, label_to_id, label_all_tokens, labels_in_gaps
):
    tokenized_inputs = tokenizer(
        text=[e[src_lang].split() for e in examples["translation"]],
        text_pair=[e[tgt_lang].split() for e in examples["translation"]],
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    tokenized_inputs["length_source"] = [len(label_src) for label_src in examples["src_tags"]]
    labels_word = []
    labels_sent = []
    for i, (hter, label_src, label_tgt) in enumerate(zip(examples["hter"], examples["src_tags"], examples["mt_tags"])):
        # remove the labels for GAPS in target
        if labels_in_gaps:
            label_tgt = [l for j, l in enumerate(label_tgt) if j % 2 != 0]
        label = label_src + label_tgt
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels_word.append(label_ids)
        labels_sent.append(hter)
    tokenized_inputs["labels"] = labels_word
    tokenized_inputs["score_sent"] = labels_sent
    return tokenized_inputs


def preprocess_sentencelevel(examples, src_lang, tgt_lang, label_column_name, tokenizer, padding):
    tokenized_inputs = tokenizer(
        text=[e[src_lang] for e in examples["translation"]],
        text_pair=[e[tgt_lang] for e in examples["translation"]],
        padding=padding,
        truncation=True,
    )
    tokenized_inputs["labels"] = examples[label_column_name]
    return tokenized_inputs
