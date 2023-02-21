
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

    
def train(model, optimizer, scheduler, train_dataloader, val_dataloader, config: Config):
    total_t0 = time.time()
    training_stats = []
    model.to(config.device)


    for epoch_i in range(0, config.num_epoch):
        # ========================================
        #               Training
        # ========================================
        print("---"*20)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print("---"*20)
        print('Training...')
        t0 = time.time()
        total_train_loss = 0.0
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(config.device)
            b_labels = batch[0].to(config.device)
            b_masks = batch[1].to(config.device)

            model.zero_grad()
            outputs = model(b_input_ids, labels=b_labels,
                            attention_mask=b_masks, token_type_ids=None)
            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss

            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                model.eval()

                sample_outputs = model.generate(bos_token_id=random.randint(1, 30000),
                                                do_sample=True,
                                                top_k=50,
                                                max_length=200,
                                                top_p=0.95,
                                                num_return_sequences=1)
                
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                
                model.train()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
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
        model.eval()
        total_eval_loss = 0.0
        nb_eval_steps = 0

        for batch in valid_data_loader:
            b_input_ids = batch[0].to(config.device)
            b_labels = batch[0].to(config.device)
            b_masks = batch[1].to(config.device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                attention_mask=b_masks,
                                labels=b_labels)

                loss = outputs[0]
            
            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss/len(val_dataloader)

        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )