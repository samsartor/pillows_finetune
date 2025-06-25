import typer
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from random import shuffle, choice, sample
from tqdm.auto import tqdm

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

DEFAULT_DATASET_PATH = './outputs/words_vs_group'
DEFAULT_CHECKPOINT = './outputs/pillow_embedding'

@app.command()
def score(
    checkpoint: list[str] = [DEFAULT_CHECKPOINT],
    words: str | None = None,
    group: str | None = None,
):
    from sentence_transformers import SentenceTransformer
    
    models = [SentenceTransformer(c) for c in checkpoint]
    if words is not None and group is not None:
        for model in models:
            query_embeddings = model.encode([group], prompt_name="query")
            document_embeddings = model.encode([words])
            similarity = model.similarity(query_embeddings, document_embeddings)
            print(similarity.item())
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
            for model in models:
                query_embeddings = model.encode([this_group], prompt_name="query")
                document_embeddings = model.encode([this_words])
                similarity = model.similarity(query_embeddings, document_embeddings)
                print(similarity.item())

@app.command()
def train(
    base_model: str = 'Qwen/Qwen3-Embedding-0.6B',
    checkpoint: str = DEFAULT_CHECKPOINT,
    dataset_path: str = DEFAULT_DATASET_PATH,
):
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import ContrastiveLoss
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers

    model = SentenceTransformer(base_model)
    loss = ContrastiveLoss(model)
    train_dataset = Dataset.load_from_disk(dataset_path)
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

@app.command()
def make_dataset(
    dataset_path: str = DEFAULT_DATASET_PATH, 
):
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

    Dataset.from_dict(dataset).save_to_disk(dataset_path)

@app.command()
def show_dataset(
    dataset_path: str = DEFAULT_DATASET_PATH,
):
    for row in Dataset.load_from_disk(dataset_path):
        print(row)

if __name__ == "__main__":
    app()
