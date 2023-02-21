from utils.lib import *


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

@dataclass
class Config():
    num_epoch: int
    batch_size: int
    warmup_steps: int
    early_stop: int
    checkpoint_dir: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id: str = 'gpt2'
    start_epoch: int = 0
    learning_rate: float = 5e-4
    epsilon: float = 1e-8


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config: Config):
        # Dictionary config
        self.config = config
        self.device = config.device
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        training_stats = []
        total_t0 = time.time()
        not_improved_count = 0
        best_valid_loss = 9999999
        is_best = False
        for epoch_i in range(self.config.start_epoch, self.config.num_epoch + 1):
            stat = self._train_epoch(epoch_i)
            training_stats.append(stat)
            if stat['Valid Loss'] < best_valid_loss:
                is_best = True
                best_valid_loss = stat['Valid Loss']
                not_improved_count = 0
            else:
                is_best = False
                not_improved_count += 1
        
            if not_improved_count > self.config.early_stop:
                print(f"Validation performance didn't improve for {self.config.early_stop} epochs")
                os.kill(os.getppid(), signal.SIGTERM)

            if is_best:
                self._save_checkpoint(epoch_i)
        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    def _save_checkpoint(self, epoch):

        filename = str(self.config.checkpoint_dir /
                       f'checkpoint-epoch{epoch}.pth')
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, filename)
        print(f"Model checkpoint saved !!!")

    def _resume_checkpoint(self, resume_path):
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint {resume_path}")
            checkpoint = torch.load(resume_path)
            self.config.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(
                f"Loaded checkpoint {resume_path} at {self.config.start_epoch}")
        else:
            print(f"No checkpoint found at {resume_path}")


class GPTTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, sample_every, train_dataloader, 
                 valid_dataloader=None, tokenizer=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.sample_every = sample_every

    def _train_epoch(self, epoch_i):
        # ========================================
        #               Training
        # ========================================
        print("---"*20)
        print('============================ Epoch {:} / {:} ============================='.format(epoch_i + 1, self.config.num_epoch))
        print("---"*20)
        print('Training...')
        t0 = time.time()
        total_train_loss = 0.0
        self.model.train()
        for step, batch in enumerate(self.train_dataloader):
            b_input_ids = batch[0].to(self.config.device)
            b_labels = batch[0].to(self.config.device)
            b_masks = batch[1].to(self.config.device)

            self.model.zero_grad()
            print(torch.max(b_input_ids).item())
            outputs = self.model(b_input_ids, labels=b_labels,
                            attention_mask=b_masks, token_type_ids=None)
            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss

            if step % self.sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(self.train_dataloader), batch_loss, elapsed))

                self.model.eval()

                sample_outputs = self.model.generate(bos_token_id=random.randint(1, 30000),
                                                do_sample=True,
                                                top_k=50,
                                                max_length=200,
                                                top_p=0.95,
                                                num_return_sequences=1)
                
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, self.tokenizer.decode(sample_output, skip_special_tokens=True)))
                
                self.model.train()

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

        avg_train_loss = total_train_loss / len(self.train_dataloader)
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()
        self.model.eval()
        total_eval_loss = 0.0
        nb_eval_steps = 0

        for batch in self.valid_dataloader:
            b_input_ids = batch[0].to(self.config.device)
            b_labels = batch[0].to(self.config.device)
            b_masks = batch[1].to(self.config.device)

            with torch.no_grad():
                outputs = self.model(b_input_ids,
                                    attention_mask=b_masks,
                                    labels=b_labels)

                loss = outputs[0]
            
            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss/len(self.valid_dataloader)

        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        return {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
                }
        




