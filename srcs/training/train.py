from utils.lib import *
from trainer import Config, GPTTrainer
from dataset import GPT2Dataset


HOME_PATH = "../../"
with initialize(config_path=r"../../config/"):
    data_cfg = compose(config_name="data-path.yaml")
data_cfg = OmegaConf.create(data_cfg)


@hydra.main(config_path='../../config', config_name='model-hyperparams')
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h_config = Config(device=device, 
                    model_id=config.gpt_2_hparams.model_id,
                    num_epoch=int(config.gpt_2_hparams.epochs),
                    start_epoch=config.gpt_2_hparams.start_epoch,
                    batch_size=config.gpt_2_hparams.batch_size,
                    learning_rate=config.gpt_2_hparams.learning_rate,
                    epsilon=config.gpt_2_hparams.epsilon,
                    warmup_steps=config.gpt_2_hparams.warmup_steps, 
                    early_stop=config.gpt_2_hparams.early_stop,
                    checkpoint_dir=config.gpt_2_hparams.checkpoint_dir)

    print(h_config.model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(h_config.model_id)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
    # print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))

    # remember to return a list of text
    text = pd.read_csv(hydra.utils.to_absolute_path(data_cfg.hblab_wiki.interim_small))['content'].tolist()
    dataset = GPT2Dataset(text, tokenizer, max_length=768)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    train_dataloader = DataLoader(train_dataset, 
                                sampler=RandomSampler(train_dataset),
                                batch_size=h_config.batch_size)

    val_dataloader = DataLoader(val_dataset,
                                sampler=RandomSampler(val_dataset),
                                batch_size=h_config.batch_size)
    total_steps = len(train_dataloader) * h_config.num_epoch


    
    
    model_config = GPT2Config.from_pretrained(h_config.model_id, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(h_config.model_id, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(),
                    lr=h_config.learning_rate,
                    eps=h_config.epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=h_config.warmup_steps,
                                                num_training_steps=total_steps)

    
    sample_every = 100
    gpt_trainer = GPTTrainer(model=model, 
                            criterion=None, 
                            metric_ftns=None, 
                            optimizer=optimizer,
                            config = h_config,
                            sample_every=sample_every,
                            train_dataloader=train_dataloader,
                            valid_dataloader = val_dataloader,
                            tokenizer=tokenizer,
                            lr_scheduler=scheduler)


    gpt_trainer.train()


if __name__ == "__main__":
    # print(data_cfg.hblab_wiki.interim)
    main()



    


