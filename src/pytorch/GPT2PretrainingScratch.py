from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import AdamW
from transformers import get_scheduler

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
padding_token_id = tokenizer.pad_token_id
# tokenizer.enable_truncation(max_length=1024)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
for k, v in model.named_parameters():
    v.requires_grad = True
optimizer = AdamW(model.parameters(), lr=5e-5)


class MainichiDataset(Dataset):
    def __init__(self):
        mainichi_path = Path('../../data/mainichi/contents.txt.shuffle')
        with mainichi_path.open('r') as f:
            train_args = f.readlines()
        train_args = [text.strip() for text in train_args]
        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        # tokenizer.enable_truncation(max_length=1024)
        train_args = [text for text in train_args if len(text) < 1024]
        train_ids = tokenizer(train_args, padding="max_length", truncation=True, add_special_tokens=True)
        train_ids
        self.data = train_ids
        # self.labels = train_ids

    def __getitem__(self, index):
        return self.data.data['input_ids'][index], self.data.data['attention_mask'][index], self.data.data['input_ids'][index]
        # return self.data.data['input_ids'][index]

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    batch_ids = [torch.LongTensor(b) for b in batch]
    batch_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True, padding_value=padding_token_id) # padding_token_id = 3

    batch_mask = [torch.LongTensor([1] * len(b)) for b in batch]
    batch_mask = torch.nn.utils.rnn.pad_sequence(batch_mask, batch_first=True, padding_value=0) # padding_mask = 0

    return {'input_ids': batch_ids, 'attention_mask': batch_mask, 'labels': batch_ids}

mainichi_train_dataset = MainichiDataset()


# train_dataloader = DataLoader(mainichi_train_dataset.data,
#                               shuffle=True,
#                               batch_size=8,
#                               collate_fn=collate_fn)
train_dataloader = DataLoader(mainichi_train_dataset.data.data['input_ids'], shuffle=True, batch_size=4, collate_fn=collate_fn)

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# from tqdm.auto import tqdm
# progress_bar = tqdm(range(num_training_steps))

info_path = Path('../../results/gpt2-mainichi/info.txt')
info_path.touch(exist_ok=True)

model.train()
for epoch in range(num_epochs):
    losses = []
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # progress_bar.update(1)
        losses.append(loss.item())
        # del loss
        if str(device) != 'cpu':
            torch.cuda.empty_cache()
    print(sum(losses))
    model_path = Path(f'../../results/gpt2-mainichi/epoch{epoch}.pkl')
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.to('cpu').state_dict(), model_path)
    model.to(device)

    with info_path.open('a') as f:
        f.write(f'epoch {epoch}: {sum(losses)}\n')


