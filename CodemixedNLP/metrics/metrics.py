import jsonlines
import math
from collections import Counter


def get_metrics(dataset_path: str, metrics: list):
    if not metrics:
        print("Metrics list cannot be empty")
        return

    result = dict()
    all_metrics = AllMetrics()
    data_reader = jsonlines.open(dataset_path)
    data = []
    for sample in data_reader:
        data.append(sample)

    for metric in metrics:
        result[metric] = getattr(all_metrics, metric)(data)
    print(result)


class AllMetrics:
    def cmi(self, data: list):
        # This way we get 12 for pos fg . GLUECos paper reports 68
        cmi = 0
        for sample in data:
            langids = sample["langids"].split(" ")
            langid_counts = Counter(langids)
            matrix_lang = langid_counts.most_common(1)[0][0]
            if len(langids) != langid_counts["other"]:
                cmi += 100*(1 - (langid_counts[matrix_lang]/(len(langids) - langid_counts["other"])))
        return cmi/len(data)

        # This was we get 52 for pos_fg
        # langids = []
        # for sample in data:
        #     langids.extend(sample["langids"].split(" "))
        # langid_counts = Counter(langids)
        # matrix_lang = langid_counts.most_common(1)[0][0]
        # return 100*(1 - (langid_counts["hi"]/(len(langids) - langid_counts["other"])))

    def m_index(self, data: list):
        m_index = 0
        for sample in data:
            langid_counts = Counter(sample["langids"].split(" "))
            denom = langid_counts["hi"] + langid_counts["en"]
            if denom == 0:
                continue

            if "other" in set(langid_counts.keys()):
                num_langs = len(langid_counts)-1
            else:
                num_langs = len(langid_counts)

            p_hi = langid_counts["hi"] / denom
            p_en = langid_counts["en"] / denom
            prob_square_sum = p_hi*p_hi + p_en*p_en

            m_index += (1 - prob_square_sum) / (num_langs*prob_square_sum)
        return m_index/len(data)

    def i_index(self, data: list):
        lang_ind_tokens = 0
        swtich = 0
        for sample in data:
            langids = sample["langids"].split(" ")
            f_langids = list(filter(lambda i: i != "other", langids))

            if len(f_langids) == 0:
                continue

            lang_ind_tokens += len(f_langids)
            prev = f_langids[0]
            for langid in f_langids[1:]:
                if langid != prev:
                    swtich += 1
                    prev = langid
        return swtich/lang_ind_tokens

    def le(self, data: list):
        le = 0
        for sample in data:
            langid_counts = Counter(sample["langids"].split(" "))
            denom = langid_counts["hi"] + langid_counts["en"]
            if denom == 0:
                continue

            sum = 0
            p_hi = langid_counts["hi"]/denom
            if p_hi != 0:
                sum += p_hi*math.log(p_hi, 2)
            p_en = langid_counts["en"]/denom
            if p_en != 0:
                sum += (p_en*math.log(p_en, 2))

            le += -1 * sum
        return le/len(data)

    def sp_avg(self, data: list):
        sp_avg = 0
        for sample in data:
            langids = sample["langids"].split(" ")
            f_langids = list(filter(lambda i: i != "other", langids))

            if len(f_langids) == 0:
                continue

            prev = f_langids[0]
            swtiches = 0
            for langid in f_langids[1:]:
                if langid != prev:
                    swtiches += 1
                    prev = langid
            sp_avg += swtiches/len(f_langids)
        return sp_avg/len(data)

    def se(self, data: list):
        raise NotImplementedError
