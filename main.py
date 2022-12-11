import streamlit as sl
from transformers import pipeline, set_seed


def is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


@sl.cache(suppress_st_warning=True, allow_output_mutation=True)
def init():
    return pipeline('text-generation', model='gpt2')


GENERATOR = init()


def generate_text(input_text, seed, max_len, num_return_seq):
    if is_int(seed):
        seed = abs(int(seed))
        if seed > 2 ** 32 - 1:
            set_seed(42)
        else:
            set_seed(seed)
    else:
        set_seed(42)

    if is_int(max_len):
        max_len = abs(int(max_len))
    else:
        max_len = 30

    if is_int(num_return_seq):
        num_return_seq = abs(int(num_return_seq))
    else:
        num_return_seq = 5

    return GENERATOR(input_text,
                     max_length=max_len,
                     num_return_sequences=num_return_seq)


def prefs():
    seed = sl.text_input("Input seed (an integer between 1 and 2**32-1)")
    max_length = sl.text_input("Input maximum number of words of the text")
    num_return_sequence = sl.text_input(
        "Input the number of desired generations")
    return seed, max_length, num_return_sequence


def main():
    input_text = sl.text_area("Input your text")

    with sl.sidebar:
        sl.header("Preferences")
        seed, max_length, num_return_sequence = prefs()
        sl.header("Current:")

        sl.caption("Seed:")
        if is_int(seed):
            if int(seed) > 2 ** 32 - 1:
                sl.write('42')
            else:
                sl.write(str(abs(int(seed))))
        else:
            sl.write('42')

        sl.caption("String length:")
        if is_int(max_length):
            sl.write(str(abs(int(max_length))))
        else:
            sl.write('30')

        sl.caption("Number of generations:")
        if is_int(num_return_sequence):
            sl.write(str(abs(int(num_return_sequence))))
        else:
            sl.write('5')

    if sl.button("Generate"):
        sl.caption("Generating...")

        result = generate_text(
            input_text, seed, max_length, num_return_sequence)

        sl.caption("Generation result:")

        i = 1
        for res in result:
            sl.caption(i)
            sl.write(res['generated_text'])
            i += 1


if __name__ == "__main__":
    main()
