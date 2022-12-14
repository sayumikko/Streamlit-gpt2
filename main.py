import streamlit as sl
from transformers import pipeline, set_seed


@sl.cache(suppress_st_warning=True, allow_output_mutation=True)
def init():
    return pipeline("text-generation", model="gpt2")


def generate_text(
    input_text,
    seed,
    max_len,
    min_len,
    num_return_seq,
    num_beams,
    top_p,
    top_k,
    do_sample,
):
    set_seed(seed)

    generator = init()

    return generator(
        input_text,
        max_length=max_len,
        num_return_sequences=num_return_seq,
        min_length=min_len,
        num_beams=num_beams,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
    )


def prefs():
    seed = sl.number_input(
        label="Input seed (an integer between 1 and 2**32-1)",
        min_value=0,
        max_value=2**32 - 1,
        value=42,
    )
    max_length = sl.number_input(
        label="Input maximum number of words of the text", min_value=0, value=30
    )
    min_length = sl.number_input(
        label="Input minimum number of words of the text",
        min_value=0,
        max_value=max_length,
        value=10,
    )
    num_return_sequence = sl.number_input(
        label="Input the number of desired generations", min_value=0, value=5
    )
    num_beams = sl.number_input(
        label="Input the size of beam search, a bigger value \
            gives more accurate, but strict text. 1 means no beam search",
        min_value=1,
        value=1,
    )
    top_k = sl.number_input(
        label="Input the number of highest probability vocabulary tokens \
             to keep for top-k-filtering.",
        value=50,
    )
    top_p = sl.number_input(label="Input the top-p-filtering.", value=1.0, step=0.1)
    do_sample = sl.checkbox("Do you want to do sample? These make the text more alive.")
    return (
        seed,
        max_length,
        min_length,
        num_return_sequence,
        num_beams,
        top_p,
        top_k,
        do_sample,
    )


def main():
    input_text = sl.text_area("Input your text")

    with sl.sidebar:
        sl.header("Preferences")
        (
            seed,
            max_length,
            min_length,
            num_return_sequence,
            num_beams,
            top_p,
            top_k,
            do_sample,
        ) = prefs()

    if sl.button("Generate"):
        sl.caption("Generating...")

        result = generate_text(
            input_text,
            seed,
            max_length,
            min_length,
            num_return_sequence,
            num_beams,
            top_p,
            top_k,
            do_sample,
        )

        sl.caption("Generation result:")

        for i, res in enumerate(result):
            sl.caption(i + 1)
            sl.write(res["generated_text"])


if __name__ == "__main__":
    main()
