import csv

import valohai

# Snippet from https://valohai-ecosystem-datasets.s3.eu-west-1.amazonaws.com/yelp_reviews_batch_inference.txt
EXAMPLE_DATA = """
Old school.....traditional "mom 'n pop" quality and perfection.
A great out of the way, non-corporate, vestige of Americana. You will love it.
Good fish sandwich.
I always feel like I am constantly bashing breweries for their food, but in my opinion, I feel the bar is raised for places like this.
I called to complain, and the "manager" didn't even apologize!!! So frustrated. Never going back.  They seem overpriced, too.
""".strip()


def test_inference(valohai_utils_global_state, monkeypatch, tmp_path):
    monkeypatch.setenv("VH_OUTPUTS_DIR", str(tmp_path))
    input_path = tmp_path / "input.txt"
    input_path.write_text(EXAMPLE_DATA)
    valohai.prepare(
        step="huggingface-classification-inference",
        default_parameters={
            "log_frequency": 1,
            # This is an untrained model, so the results won't be very interesting.
            "huggingface_repository": "distilbert-base-uncased",
            "output_path": "test.csv",
        },
        default_inputs={
            "data": str(input_path),
        },
    )
    from models.nlp.classification.huggingface.inference import main

    main()
    with (tmp_path / "test.csv").open() as f:
        results = list(csv.DictReader(f))
    assert len(results) == 5
