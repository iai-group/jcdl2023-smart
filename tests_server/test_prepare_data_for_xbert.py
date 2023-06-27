import smart.classification.type.prepare_data_for_xbert as pdfx
import os
import shutil
import scipy.sparse as smat
import pickle


class TestPrepareDataForXbert:
    def __check_file_exists(self, file_name):
        return os.path.exists(file_name)

    def __clean_data_folder(self, data_folder):
        if not os.path.exists(data_folder):
            return
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

    def __read_id_map_file(self, file):
        id_map = {}
        with open(file) as in_file:
            for line in in_file:
                tokens = line.rstrip().split("\t")
                id_map[tokens[0]] = int(tokens[1])
        return id_map

    def __read_label_map_file(self, file):
        label_map = {}
        index = 0
        with open(file) as in_file:
            for line in in_file:
                tokens = line.rstrip().split(":")
                label_map[index] = tokens[0].rstrip()
                index = index + 1
        return label_map

    def __test_data_files_exist(self, dataset):
        """First verify that it created X.train.npz, X.test.npz, Y.train.npz,
        Y.test.npz, train_raw_texts.txt, test_raw_texts.txt, label_map.txt and
        mlb.pkl"""

        # assert dataset == "dbpedia"
        xbert_folder = pdfx.XBERT_DATA_FOLDER + "/" + dataset + "_full"
        folder = xbert_folder

        # If the data folder doesnt exist create it!
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            self.__clean_data_folder(folder)
            print("Successfully cleander folders")
        pdfx.prepare_xbert_data(dataset, use_sample=False)
        assert self.__check_file_exists(folder + "/X.train.npz")
        assert self.__check_file_exists(folder + "/X.test.npz")
        assert self.__check_file_exists(folder + "/Y.train.npz")
        assert self.__check_file_exists(folder + "/Y.test.npz")
        assert self.__check_file_exists(folder + "/train_raw_texts.txt")
        assert self.__check_file_exists(folder + "/test_raw_texts.txt")
        assert self.__check_file_exists(folder + "/label_map.txt")
        assert self.__check_file_exists(folder + "/mlb.pkl")
        assert self.__check_file_exists(folder + "/train_id_map.txt")
        assert self.__check_file_exists(folder + "/test_id_map.txt")

    def test_data_files_exist(self):
        self.__test_data_files_exist("dbpedia")
        self.__test_data_files_exist("wikidata")

    def test_data_correctness(self):
        self.__test_data_correctness("dbpedia")
        self.__test_data_correctness("wikidata")

    def __test_data_correctness(self, dataset):
        """Check if the data shapes and label info are all consistent."""

        folder = pdfx.XBERT_DATA_FOLDER + "/" + dataset + "_full"
        # If the data folder doesnt exist create it!
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            self.__clean_data_folder(folder)
        test_data_df, train_data_df = pdfx.prepare_xbert_data(
            dataset,
            use_sample=False,
            use_clean_data=False,
            use_entity_descriptions=True,
        )

        train_id_map = self.__read_id_map_file(folder + "/train_id_map.txt")
        test_id_map = self.__read_id_map_file(folder + "/test_id_map.txt")
        label_map = self.__read_label_map_file(folder + "/label_map.txt")

        X_train = smat.load_npz(folder + "/X.train.npz")
        X_test = smat.load_npz(folder + "/X.test.npz")
        Y_train = smat.load_npz(folder + "/Y.train.npz")
        Y_test = smat.load_npz(folder + "/Y.test.npz")

        # Test if int labels match string labels
        mlb = pickle.load(open(folder + "/mlb.pkl", "rb"))

        # Test if the shapes of the sample data matches
        assert (
            X_train.shape[0] == train_data_df.shape[0]
            and X_train.shape[1] == train_data_df.iloc[0].TFIDF.shape[1]
        )
        assert (
            X_test.shape[0] == test_data_df.shape[0]
            and X_test.shape[1] == train_data_df.iloc[0].TFIDF.shape[1]
        )
        assert Y_train.shape[0] == train_data_df.shape[0] and Y_train.shape[
            1
        ] == len(train_data_df.iloc[0].labels)
        assert Y_test.shape[0] == test_data_df.shape[0] and Y_test.shape[
            1
        ] == len(train_data_df.iloc[0].labels)
        self.__verify_labels(
            Y_train, label_map, train_data_df, train_id_map, mlb, dataset
        )
        self.__verify_labels(
            Y_test, label_map, test_data_df, test_id_map, mlb, dataset
        )

        self.__verify_question_string(
            folder + "/train_raw_texts.txt",
            train_data_df,
            train_id_map,
            dataset,
        )
        self.__verify_question_string(
            folder + "/test_raw_texts.txt", test_data_df, test_id_map, dataset
        )

    def __verify_question_string(self, file, data_df, id_map, dataset):
        with open(file) as train_q_file:
            for q_id, q_line in zip(id_map, train_q_file):
                # Wikidata has int ids
                if dataset == "wikidata":
                    q_id = int(q_id)
                df_row = data_df[data_df["id"] == q_id]
                question = df_row["question"].values[0]
                assert question.strip() == q_line.strip()

    def __verify_labels(
        self, Y_train, label_map, train_data_df, id_map, mlb, dataset
    ):
        for q_id in id_map:
            row_num = id_map[q_id]
            print(row_num, q_id)
            labels = Y_train[row_num].nonzero()[1]

            # Wikidata has int ids
            if dataset == "wikidata":
                q_id = int(q_id)

            df_row = train_data_df[train_data_df["id"] == q_id]
            assert len(labels) == len(df_row["type"].values.tolist()[0])
            label_set = set(df_row["type"].values.tolist()[0])
            print(labels)
            # check if the label string given to xbert matches
            for label in labels:
                label_str = label_map[int(label)]
                # DBPedia types have dbo prefixes
                if dataset == "dbpedia":
                    label_str = "dbo:" + label_str
                print(q_id, label_str, label)
                assert label_str in label_set

            # Check sanity of MultiLabelBinarizer
            labels_str = mlb.inverse_transform(Y_train[row_num])[0]
            for label_str in labels_str:
                assert label_str in label_set
