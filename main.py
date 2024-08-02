from transformers import pipeline

classifier_1 = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
classifier_2 = pipeline("zero-shot-classification",
                      model="sileod/deberta-v3-large-tasksource-nli")
classifier_3 = pipeline("zero-shot-classification",
                      model="sileod/mdeberta-v3-base-tasksource-nli")
classifier_4 = pipeline("zero-shot-classification",
                      model="AyoubChLin/DistilBERT_ZeroShot")
classifier_5 = pipeline("zero-shot-classification",
                      model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

labels = ["small talk", "flirting", "casual conversation", "sexual discussion", "neutral"]

with open('input') as fp:
    while True:
        line = fp.readline().replace("\n", "")
        guesses = [
            classifier_1(line, labels, multi_label=True)['labels'][0],
            classifier_2(line, labels, multi_label=True)['labels'][0],
            classifier_3(line, labels, multi_label=True)['labels'][0],
            classifier_4(line, labels, multi_label=True)['labels'][0],
            classifier_5(line, labels, multi_label=True)['labels'][0]
        ]

        points = {}
        for guess in guesses:
            try:
                points[guess] += 1
            except KeyError:
                points[guess] = 1

        rating = max(points, key=points.get).replace("\n", "").lower()

        with open("output.csv", "a") as out:
            out.write(line + "," + rating + "\n")
