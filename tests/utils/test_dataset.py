"""Tests for Dataset class."""

from smart.utils.dataset import Dataset


class TestDataset:
    def test_dbpedia_train(self):
        queries = Dataset("dbpedia", "train").get_queries()
        # Total number of queries minus empty question text minus duplicate
        # IDs minus n/a questions.
        assert len(queries) == 17198
        # Question with unmodified text.
        assert (
            queries["dbpedia_23480"]
            == "Do Prince Harry and Prince William have the same parents?"
        )
        # Question with modified text.
        assert (
            queries["dbpedia_11196"]
            == "Who is famous for of writers of To the Christian Nobility of "
            "the German Nation?"
        )
        # Question with empty text.
        assert "dbpedia_9619" not in queries
        # Question with n/a text
        assert "dbpedia_6488" not in queries

    def test_valid_question(self):
        dataset = Dataset("dbpedia", "test")
        question = "na"
        assert not dataset.is_valid_question(question_text=question)
        question = "n/a"
        assert not dataset.is_valid_question(question_text=question)
        question = "null"
        assert not dataset.is_valid_question(question_text=question)
        question = "Who is?"
        assert not dataset.is_valid_question(question_text=question)

    def test_dbpedia_expanded_types(self):
        dataset = Dataset("dbpedia", "train", expand_types=True)
        data_df = dataset.get_df()
        expanded_types_gt = [
            'dbo:Opera',
            'dbo:MusicalWork',
            'dbo:Work',
            'dbo:Database',
            'dbo:LineOfFashion',
            'dbo:Document',
            'dbo:MusicalWork',
            'dbo:Artwork',
            'dbo:Website',
            'dbo:CollectionOfValuables',
            'dbo:TelevisionEpisode',
            'dbo:TelevisionShow',
            'dbo:Cartoon',
            'dbo:Software',
            'dbo:TelevisionSeason',
            'dbo:WrittenWork',
            'dbo:RadioProgram',
            'dbo:Work',
            'dbo:Film',
        ]
        expanded_types = data_df[data_df["id"] == "dbpedia_14427"][
            "type"
        ].values[0]
        expanded_types_set = set(expanded_types)
        for expanded_type in expanded_types_gt:
            assert expanded_type in expanded_types_set
        assert len(expanded_types) == 17
