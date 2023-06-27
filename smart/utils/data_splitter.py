from smart.utils.dataset import Dataset


# def fix_types(data: str, split: str) -> None:
#     dataset_original = Dataset(data, split)
#     dataset_clean = Dataset(data, f"{split}_clean_grammarly")
#     for data in dataset_original.dataset:


def join_data():
    all_data = []
    total_count = 0
    data = "wikidata"
    split = "train"
    data_train_clean = Dataset(data, split)
    for i in range(37):
        filename = f"splits/{data}_{split}_split{i}.json"
        print(filename)
        dataset = Dataset(data, split)
        dataset._load_data(filename)
        total_count += len(dataset._data)
        print(len(dataset._data))
        all_data.extend(dataset._data)
    print(len(all_data), total_count, len(data_train_clean._data))
    data_train_clean.dump_json(all_data, f"{data}_{split}_clean_grammarly.json")


def split_data():
    data = "wikidata"
    split = "test"
    data_train = Dataset(data, split)
    num_queries = len(data_train._data)
    for i in range(int(num_queries / 500.0) + 1):
        print(i * 500, (i + 1) * 500)
        data_train.dump_json(
            data_train._data,
            f"splits/{data}_{split}_split{i}.json",
            i * 500,
            (i + 1) * 500,
        )


if __name__ == "__main__":
    # split_data()
    join_data()
    # data_train = Dataset("dbpedia", "train_clean")

    # for i in range(87):
    #     print(i * 500, (i + 1) * 500)
    #     data_train.dump_json(
    #         data_train._data, f"splits/split{i}.json", i * 500, (i + 1) * 500
    #     )
    # data_test = Dataset("dbpedia", "test")
    # data_test_clean = Dataset("dbpedia", "test_clean_grammar")
    # missing = []

    # for data1 in data_test._data:
    #     found = False
    #     for data2 in data_test_clean._data:
    #         if data1["id"] == data2["id"]:
    #             found = True
    #     if not found:
    #         print(data1["id"])
    #         missing.append(data1)
    # Dataset.dump_json(missing, "data/smart_dataset/dbpedia/missing.json",
    # 0, 500)
