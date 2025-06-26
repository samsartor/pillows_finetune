import typer
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from random import shuffle, choice, sample
from tqdm.auto import tqdm
from pathlib import Path
import requests

app = typer.Typer()

@dataclass
class Group:
    level: int
    name: str
    words: list[str]
    score: float = 1
    ignore: bool = False

    @property
    def fmt_words(self):
        return ', '.join(map(lambda w: w.lower(), self.words))

    @property
    def fmt_group(self):
        return f'{self.name.lower().replace("___", "_")}'

DEFAULT_CHECKPOINT = './outputs/pillow_embedding'
DEFAULT_WORD_LIST_URL = 'https://github.com/first20hours/google-10000-english/raw/refs/heads/master/google-10000-english-no-swears.txt'
DEFAULT_WORD_LIST_PATH = './outputs/word_list.txt'

class Scorer:
    def __init__(self, checkpoint: str, normalize: int, shuffle: int):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(checkpoint)
        if normalize > 0:
            words = word_list()
            words = [
                ', '.join(sample(words, k=4))
                for _ in range(normalize)
            ]
            self.rand_embeddings = self.model.encode(words)
        else:
            self.rand_embeddings = None
        self.shuffle = shuffle

    def score(self, group: str, words: str) -> float:
        to_score = []
        if self.shuffle > 0:
            split_words = list(map(lambda w: w.strip(), words.split(',')))
            for _ in range(self.shuffle):
                shuffle(split_words)
                to_score.append(', '.join(split_words))
        else:
            to_score.append(words)
        group_embeddings = self.model.encode([group], prompt_name="query")
        word_embeddings = self.model.encode(to_score)
        word_similarity = self.model.similarity(group_embeddings, word_embeddings)
        if self.rand_embeddings is None:
            return word_similarity.mean().item()
        else:
            word_similarity = word_similarity.log().mean()
            rand_similarity = self.model.similarity(group_embeddings, self.rand_embeddings).log().mean()
            return (word_similarity - rand_similarity).item()

@app.command()
def score(
    checkpoint: list[str] = [DEFAULT_CHECKPOINT],
    words: str | None = None,
    group: str | None = None,
    normalize: int = 10,
    shuffle: int = 4,
):
    from sentence_transformers import SentenceTransformer
    
    scorers = [Scorer(c, normalize=normalize, shuffle=shuffle) for c in checkpoint]
    if words is not None and group is not None:
        for scorer in scorers:
            print(scorer.score(group, words))
    if words is None or group is None:
        while True:
            if group is None:
                this_group = input('Group (eg "eat voraciously"): ')
            else:
                this_group = group
            if words is None:
                this_words = input('Words (eg "wolf, gulp, gobble, scarf"): ')
            else:
                this_words = words
            for scorer in scorers:
                print(scorer.score(this_group, this_words))

@app.command()
def train(
    base_model: str = 'Qwen/Qwen3-Embedding-0.6B',
    checkpoint: str = DEFAULT_CHECKPOINT,
):
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import ContrastiveLoss
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers

    model = SentenceTransformer(base_model)
    loss = ContrastiveLoss(model)
    train_dataset = make_dataset()
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=checkpoint,
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if your GPU can't handle FP16
        bf16=False,  # Set to True if your GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(checkpoint)

def make_dataset() -> Dataset:
    groups: dict[tuple[int, int], Group] = {}
    for row in tqdm(load_dataset("eric27n/NYT-Connections", split="train"), 'Scanning dataset'):
        id = row['Game ID']
        level = row['Group Level']
        if (id, level) not in groups:
            groups[(id, level)] = Group(level=level, name=row['Group Name'], words=[])            
        if not isinstance(row['Word'], str):
            groups[(id, level)].ignore = True
        groups[(id, level)].words.append(row['Word'])

    dataset = {'sentence1': [], 'sentence2': [], 'score': []}
    for ((id, level), correct) in tqdm(groups.items(), 'Making groups'):
        if correct.ignore:
            continue
        
        shuffle(correct.words)

        other_words = []
        other_names = []
        for l in (0, 1, 2, 3):
            if l != level:
                other_group = groups[(id, l)]
                if other_group.ignore:
                    continue
                other_names.append(other_group.name)
                for w in other_group.words:
                    other_words.append(w)

        incorrect = Group(
            level=level,
            name=correct.name,
            words=sample(other_words, k=4),
            score=0,
        )

        for group in [correct, incorrect]:
            dataset['sentence1'].append(group.fmt_group)
            dataset['sentence2'].append(group.fmt_words)
            dataset['score'].append(group.score)

    return Dataset.from_dict(dataset)

def word_list() -> list[str]:
    path = Path(DEFAULT_WORD_LIST_PATH)
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        response = requests.get(DEFAULT_WORD_LIST_URL)
        with path.open('w') as f:
            f.write(response.text)
    return path.read_text().split()[:1000]

@app.command()
def show_dataset():
    for row in make_dataset():
        print(row)

@app.command()
def random_words():
    print(', '.join(sample(word_list(), k=4)))

if __name__ == "__main__":
    app()
