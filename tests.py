import CodemixedNLP as csnlp

DATASET_FOLDER = "./datasets"


def test_bert_only_classification(pretrained_name_or_path, dataset="dummysail2017/Hinglish"):
    dataset_folder = f"{DATASET_FOLDER}/{dataset}"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders="bert",
        encoder_attributes={"pretrained_name_or_path": pretrained_name_or_path},
        task_attributes={"name": "classification"},
        target_label_fields="label",
        mode="train",
        max_epochs=1)
    return


def test_bert_only_classification_tagging(pretrained_name_or_path):
    dataset_folder = f"{DATASET_FOLDER}/dummysail2017/Hinglish"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders="bert",
        encoder_attributes={"pretrained_name_or_path": pretrained_name_or_path},
        task_attributes=[{"name": "classification"}, {"name": "seq_tagging"}],
        target_label_fields=["label", "langids"],
        mode="train")
    return


def test_bert_only_tagging(pretrained_name_or_path, label_field):
    dataset_folder = f"{DATASET_FOLDER}/dummysail2017/Hinglish"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders="bert",
        encoder_attributes={"pretrained_name_or_path": pretrained_name_or_path},
        task_attributes={"name": "seq_tagging"},
        target_label_fields=label_field,
        mode="train")
    return


def test_bert_plus_lstm_classification(pretrained_name_or_path,
                                       lstm_input_representation,
                                       encodings_merge_type="concat"):
    dataset_folder = f"{DATASET_FOLDER}/dummysail2017/Hinglish"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders=["bert", "lstm"],
        encoder_attributes=[
            {"pretrained_name_or_path": pretrained_name_or_path},
            {"input_representation": lstm_input_representation}],
        task_attributes={"name": "classification"},
        target_label_fields="label",
        mode="train",
        encodings_merge_type=encodings_merge_type,
        debug=True)
    return


def test_bert_only_classification_with_xxx(xxx_input_label_field):
    dataset_folder = f"{DATASET_FOLDER}/dummysail2017/Hinglish"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders="bert",
        encoder_attributes={
            "pretrained_name_or_path": "/Users/muralidhar/Education/CMU/11927Capstone/Codemixed/checkpoints/pretrained/bert-base-multilingual-cased"},
        task_attributes={"name": "classification"},
        target_label_fields="label",
        mode="train",
        xxx_input_label_field=xxx_input_label_field)
    return


def test_bert_only_classification_with_fusion(pretrained_name_or_path):
    dataset_folder = f"{DATASET_FOLDER}/dummysail2017/Hinglish"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders="bert",
        encoder_attributes={"pretrained_name_or_path": pretrained_name_or_path},
        task_attributes={"name": "classification"},
        target_label_fields="label",
        mode="train",
        fusion_text_fields=["text", "text_pp"])
    return


def test_bert_only_classification_from_checkpoint(pretrained_name_or_path, ckpt_path):
    dataset_folder = f"{DATASET_FOLDER}/dummysail2017/Hinglish"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders="bert",
        encoder_attributes={"pretrained_name_or_path": pretrained_name_or_path},
        task_attributes={"name": "classification"},
        target_label_fields="label",
        mode="test",
        eval_ckpt_path=ckpt_path)
    return


def test_bert_only_classification_orgdata(pretrained_name_or_path):
    dataset_folder = f"../datasets/sail2017/Hinglish"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders="bert",
        encoder_attributes={"pretrained_name_or_path": pretrained_name_or_path},
        task_attributes={"name": "classification"},
        target_label_fields="label",
        mode=["train", "test"])
    return


def test_two_berts_based_classification():
    dataset_folder = f"{DATASET_FOLDER}/dummysail2017/Hinglish"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders=["bert", "bert"],
        encoder_attributes=[{"pretrained_name_or_path": "bert-base-multilingual-cased"},
                            {"pretrained_name_or_path": "xlm-roberta-base"}],
        task_attributes={"name": "classification"},
        target_label_fields="label",
        mode="train",
        max_epochs=5,
        encodings_merge_type="weighted_ensemble")
    return


def test_bert_only_tagging_orgdata(pretrained_name_or_path):
    dataset_folder = f"../datasets/gluecos_ner/Hinglish"
    csnlp.benchmarks.run_unified(
        dataset_folder=dataset_folder,
        encoders="bert",
        encoder_attributes={"pretrained_name_or_path": pretrained_name_or_path},
        task_attributes={"name": "seq_tagging"},
        target_label_fields="nertags",
        mode=["train", "test"])
    return


if __name__ == "__main__":
    """ training (dummy datasets) """

    # TEST 1
    # test_bert_only_classification(pretrained_name_or_path="bert-base-multilingual-cased")
    # test_bert_only_classification(pretrained_name_or_path="xlm-roberta-base")

    # TEST 2
    # test_bert_only_tagging(pretrained_name_or_path="xlm-roberta-base", label_field="langids")
    # test_bert_only_classification_tagging(pretrained_name_or_path="xlm-roberta-base")

    # TEST 3
    # test_bert_plus_lstm_classification(pretrained_name_or_path="xlm-roberta-base", lstm_input_representation="sc")
    # # test_bert_plus_lstm_classification(pretrained_name_or_path="xlm-roberta-base",
    # #                                    lstm_input_representation="fasttext")

    # TEST 4
    # test_bert_only_classification_with_xxx(xxx_input_label_field="langids")

    # TEST 5
    # test_bert_only_classification_with_fusion(pretrained_name_or_path="xlm-roberta-base")

    # TOEST 6 - binary classification
    # test_bert_only_classification(pretrained_name_or_path="xlm-roberta-base",
    #                               dataset="dummyhatespeech/Hinglish")

    """ weighted ensemble concat models """

    # test_bert_plus_lstm_classification(pretrained_name_or_path="xlm-roberta-base",
    #                                    lstm_input_representation="sc",
    #                                    encodings_merge_type="weighted_ensemble")
    # test_two_berts_based_classification()

    """ from checkpoints (dummy datasets) """

    # test_bert_only_classification_from_checkpoint(
    #     pretrained_name_or_path="bert-base-multilingual-cased",
    #     ckpt_path="/Users/muralidhar/Education/CMU/11927Capstone/Codemixed/CodemixedNLP/datasets/dummysail2017/Hinglish/checkpoints/2021-03-27_06:32:02.754301"
    # )

    """ training (org. datasets) """

    # test_bert_only_classification_orgdata(pretrained_name_or_path="bert-base-multilingual-cased")
    # test_bert_only_tagging_orgdata(pretrained_name_or_path="xlm-roberta-base")

    print("complete")
