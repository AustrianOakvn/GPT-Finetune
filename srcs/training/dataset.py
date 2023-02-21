from utils.lib import *

class GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, gpt2_type="", max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dicts = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')

            self.input_ids.append(torch.tensor(encodings_dicts['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dicts['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]




if __name__ == "__main__":
    ## Init and test dataset here
    pass