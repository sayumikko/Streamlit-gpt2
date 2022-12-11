import streamlit as sl
from transformers import pipeline, set_seed


def is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


def generate(input_text, seed, max_len, num_return_seq):
    generator = pipeline('text-generation', model='gpt2')

    if is_int(seed):
        if int(seed) > 2 ** 32 - 1:
            set_seed(42)
        else:
            set_seed(int(seed))
    else:
        set_seed(42)

    sl.caption("Generating...")

    if is_int(max_len):
        max_len = int(max_len)
    else:
        max_len = 30

    if is_int(num_return_seq):
        num_return_seq = int(num_return_seq)
    else:
        num_return_seq = 5

    result = generator(input_text,
                       max_length=max_len,
                       num_return_sequences=num_return_seq)
    sl.caption("Generation result:")

    i = 1
    for res in result:
        sl.caption(i)
        sl.write(res['generated_text'])
        i += 1


def prefs():
    seed = sl.text_input("Input seed")
    max_length = sl.text_input("Input maximum number of words of the text")
    num_return_sequence = sl.text_input(
        "Input the number of desired generations")
    return seed, max_length, num_return_sequence


def gener(seed, max_len, num_return_seq):
    input_text = sl.text_area("Input your text")
    generate(input_text, seed, max_len, num_return_seq)


def main():
    sl.set_page_config(layout="wide")
    sl.title("Gpt2 text-to-text generator.")

    with sl.sidebar:
        sl.header("Preferences")
        seed, max_length, num_return_sequence = prefs()
        sl.header("Current:")
        sl.caption("Seed:")
        sl.write(seed)
        sl.caption("String length:")
        sl.write(max_length)
        sl.caption("Number of generations:")
        sl.write(num_return_sequence)

    gener(seed, max_length, num_return_sequence)


if __name__ == "__main__":
    main()
