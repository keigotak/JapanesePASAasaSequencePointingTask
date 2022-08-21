import torch
from torch_cka import CKA

from SimilarlityBERTDoubletEvaluate import get_properties, get_datasets, allocate_data_to_device

class WSCTestDataset(torch.utils.data.Dataset):
    def __init__(self):
        test_sentence1, test_sentence2, test_labels = get_datasets(
            '/clwork/keigo/JapanesePASAasaSequencePointingTask/data/Winograd-Schema-Challenge-Ja-master/fixed/test-triplet.txt')
        self.s1 = test_sentence1
        self.s2 = test_sentence2
        self.labels = test_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.s1[idx], self.s2[idx], self.labels[idx]

class Model(torch.nn.Modules):
    def __init__(self, run_mode):
        model_name, OUTPUT_PATH, NUM_EPOCHS = get_properties(run_mode)
        if 'gpt2' in model_name:
            model = AutoModel.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            tokenizer.do_lower_case = True
        elif 'mbart' in model_name:
            model = MBartForConditionalGeneration.from_pretrained(model_name)
            tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ja_XX", tgt_lang="ja_XX")
        elif 't5' in model_name:
            model = T5Model.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
        elif 'tohoku' in model_name:
            model = BertModel.from_pretrained(model_name)
            tokenizer = BertJapaneseTokenizerFast.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        weight_path = Path(OUTPUT_PATH + f'/{model_name.replace("/", ".")}.{NUM_EPOCHS}.pt')
        with weight_path.open('rb') as f:
            weights = torch.load(f)

        model.load_state_dict(weights['model'], strict=True)
        self.model = allocate_data_to_device(model, DEVICE)
        output_layer = torch.nn.Linear(model.config.hidden_size, 1)
        output_layer.load_state_dict(weights['output_layer'], strict=True)
        self.output_layer = allocate_data_to_device(output_layer, DEVICE)
        self.tokenizer = tokenizer

    def forward(self, s1, s2):
        tokens1 = self.tokenizer(s1)
        tokens2 = self.tokenizer(s2)

        x1 = self.model(**tokens1)
        x2 = self.model(**tokens2)

        x1 = average_pooling(x1, tokens1['attention_mask'])
        x2 = average_pooling(x2, tokens2['attention_mask'])

        x1 = self.output_layer(x1)
        x2 = self.output_layer(x2)
        return x1, x2

BATCH_SIZE = 32
DEVICE = 'cuda:0'  # 'cuda:0'
with_activation_function = False
with_print_logits = False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model1 = Model('rinna-gpt2')
model2 = Model('xlm-roberta-large')

dataloader = DataLoader(WSCTestDataset,
                        batch_size=BATCH_SIZE, # according to your device memory
                        shuffle=False)  # Don't forget to seed your dataloader

cka = CKA(model1, model2,
          model1_name="ResNet18",   # good idea to provide names to avoid confusion
          model2_name="ResNet34",
          model1_layers=layer_names_resnet18, # List of layers to extract features from
          model2_layers=layer_names_resnet34, # extracts all layer features by default
          device='cuda')

cka.compare(dataloader) # secondary dataloader is optional

results = cka.export()  # returns a dict that contains model names, layer names
                        # and the CKA matrix
