from smart.utils.type_description import TypeDescription


class TestTypeDescription:
    def test_dbpedia_type_description(self):
        dataset = "dbpedia"
        td = TypeDescription(dataset, use_entity_descriptions=False)
        assert td.get_type_description("dbo:AcademicJournal").replace(
            "\n", ""
        ).split(".")[0].strip() == (
            "An academic journal is a mostly peer-reviewed periodical in "
            "which scholarship relating to a particular academic discipline "
            "is published"
        )
        assert td.get_type_description("dbo:NationalAnthem").replace(
            "\n", ""
        ).strip() == (
            "Patriotic musical composition which is the offcial national "
            "song."
        )
        assert td.get_type_description("dbo:Activity") is not None

    def test_dbpedia_type_description_with_entity_abstracts(self):
        dataset = "dbpedia"
        td = TypeDescription(dataset, use_entity_descriptions=True)
        assert (
            td.get_type_description("dbo:Activity").split(".")[0].strip()
            == "event; actions that result in changes of state"
        )

    def test_wikidata_type_description(self):
        dataset = "wikidata"
        td = TypeDescription(dataset)
        assert td.get_type_description("academic journal").replace(
            "\n", ""
        ).split(".")[0].strip() == (
            "peer-reviewed periodical relating to a particular academic "
            "discipline"
        )
        assert td.get_type_description("alliance").replace("\n", "") == (
            "express agreement under international law entered into by "
            "actors in international law."
        )
