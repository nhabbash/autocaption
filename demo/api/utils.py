
def decode_cap(encoded_cap, v):
    return " ".join([v.idx2word[idx] for idx in encoded_cap if idx not in [v(v.start_word), v(v.end_word), v(v.pad_word)]])

def get_caption(img, encoder, decoder, vocab, sample="beam", decode=True):
    # Generate a caption from the image
    features = encoder(img).unsqueeze(1)

    if sample == "beam":
        sample = decoder.sample_beam_search(features)
    else:
        sample = [decoder.sample(features)]

    if decode:
        # Convert word_ids to words
        sentences = [decode_cap(s, vocab) for s in sample]
        return sentences
    else:
        return sample