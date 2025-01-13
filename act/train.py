import torch
from torch import nn
from act.utils import save_checkpoint, get_path
from tqdm import tqdm

def train(cfg, model, optimizer, scheduler, data_manager, device):
    if cfg.SAVE:
        log_path = get_path(cfg.SAVE_PATH, 'log_')
        with open(log_path, "w") as log_file:
            if not cfg.LOAD:
                log_file.write("accuracy,loss,ponder cost,steps\n")

    train_dataloader = data_manager.create_dataloader(cfg)
    test_dataloader = data_manager.create_dataloader(cfg, train=False)

    best_precision = cfg.BEST_PRECISION
    for epoch in range(cfg.START_EPOCH, cfg.SOLVER.EPOCHS):
        is_best = False

        train_epoch(epoch=epoch, cfg=cfg, model=model, dataloader=train_dataloader, optimizer=optimizer, device=device)

        avg_precision, avg_loss, avg_ponder, steps = test_epoch(cfg=cfg, model=model, dataloader=test_dataloader, device=device)

        if scheduler:
            scheduler.step(avg_precision)

        print(f"Epoch {epoch}: Precision={avg_precision:.4f}, Loss={avg_loss:.4f}, Ponder Cost={avg_ponder:.4f}, Steps={steps:.4f}")

        if avg_precision > best_precision:
            best_precision = avg_precision
            is_best = True

        if cfg.SAVE:
            if not epoch % cfg.SAVE_INTERVAL:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_precision': best_precision,
                    'precision': avg_precision,
                    'optimizer': optimizer.state_dict(),
                }, is_best, cfg)

            with open(log_path, "a") as log_file:
                log_file.write(", ".join(str(elem) for elem in (avg_precision, avg_loss, avg_ponder, steps)) + "\n")

        print(f"Epoch {epoch} completed.\n{'-' * 40}")

def train_epoch(epoch, cfg, model, dataloader, optimizer, device):
    model.train()
    if cfg.MODEL.TASK == 'parity':
        loss_func = nn.BCEWithLogitsLoss()
        iterator = tqdm(dataloader)
        iterator.set_description(f"Train Epoch {epoch}")
        for x, y in iterator:
            x = x.view(cfg.DATALOADER.BATCH_SIZE, -1, cfg.INPUT.DIM).transpose(0, 1).to(device)
            y = y.view(cfg.DATALOADER.BATCH_SIZE, cfg.INPUT.SEQ_LEN, -1).transpose(0, 1).to(device)

            state = model.init_hidden(device)
            y_pred, (_, _, ponder_cost, _) = model(x, state, cfg.ACT.HALT_TRAIN)

            loss = loss_func(y_pred, y) + (cfg.ACT.PENALTY_COEF * torch.mean(ponder_cost))

            optimizer.zero_grad()
            loss.backward()
            if cfg.SOLVER.GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)
            optimizer.step()

    elif cfg.MODEL.TASK == 'addition':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
        max_D = cfg.INPUT.DIM // 10
        iterator = tqdm(dataloader)
        iterator.set_description(f"Train Epoch {epoch}")
        for x, y, dec_y, loss_mask in iterator:
            x = x.view(cfg.DATALOADER.BATCH_SIZE, -1, cfg.INPUT.DIM).transpose(0, 1).to(device)
            y = y.view(cfg.DATALOADER.BATCH_SIZE, -1, max_D + 1).transpose(0, 1).to(device)
            loss_mask = loss_mask.to(device)

            state = model.init_hidden(device)
            y_pred, (_, _, ponder_cost, _) = model(x, state, cfg.ACT.HALT_TRAIN)

            masked_pred = y_pred[1:, :].contiguous().view(-1, 11)
            masked_tgt = y[1:, :].contiguous().view(-1)
            loss = loss_func(masked_pred, masked_tgt) + (cfg.ACT.PENALTY_COEF * torch.mean(ponder_cost))

            optimizer.zero_grad()
            loss.backward()
            if cfg.SOLVER.GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)
            optimizer.step()
    else:
        raise NotImplementedError

def norm_corrects(cfg, out, y, mask):
    mask = mask.long()
    y = y.transpose(0, 1)
    max_D = cfg.INPUT.DIM // 10
    _, indices = torch.max(out.transpose(0, 1).view(cfg.DATALOADER.BATCH_SIZE, -1, max_D + 1, 11), dim=3)
    masked_indices = torch.mul(indices, mask.unsqueeze(2).expand(indices.shape))
    masked_y = torch.mul(y, mask.unsqueeze(2).expand(y.shape))
    n_not_masked = mask.sum()
    correct_nums = (masked_indices == masked_y).all(dim=2).sum() - (mask.numel() - n_not_masked)
    return correct_nums.float() / n_not_masked

def test_epoch(cfg, model, dataloader, device):
    model.eval()
    with torch.no_grad():
        n = 0
        all_corrects = 0
        losses = 0
        ponder_costs = 0
        acc_steps = 0

        if cfg.MODEL.TASK == 'parity':
            loss_func = nn.BCEWithLogitsLoss()
            iterator = tqdm(dataloader)
            iterator.set_description("Validating")
            for n, (x, y) in enumerate(iterator):
                x = x.view(cfg.DATALOADER.BATCH_SIZE, -1, cfg.INPUT.DIM).transpose(0, 1).to(device)
                y = y.view(cfg.DATALOADER.BATCH_SIZE, cfg.INPUT.SEQ_LEN, -1).transpose(0, 1).to(device)
                state = model.init_hidden(device)
                y_pred, (_, _, ponder_cost, steps) = model(x, state, cfg.ACT.HALT_TEST)
                loss = loss_func(y_pred, y) + (cfg.ACT.PENALTY_COEF * torch.mean(ponder_cost))
                all_corrects += (y.eq(torch.round(torch.sigmoid(y_pred)))).float().mean().item()
                losses += loss.item()
                ponder_costs += torch.mean(ponder_cost).item()
                acc_steps += torch.mean(steps).item()

        elif cfg.MODEL.TASK == 'addition':
            loss_func = nn.CrossEntropyLoss(reduction='mean')
            max_D = cfg.INPUT.DIM // 10
            iterator = tqdm(dataloader)
            iterator.set_description("Validating")
            for n, (x, y, dec_y, loss_mask) in enumerate(iterator):
                x = x.view(cfg.DATALOADER.BATCH_SIZE, -1, cfg.INPUT.DIM).transpose(0, 1).to(device)
                y = y.view(cfg.DATALOADER.BATCH_SIZE, -1, max_D + 1).transpose(0, 1).to(device)
                loss_mask = loss_mask.to(device)
                state = model.init_hidden(device)
                y_pred, (_, _, ponder_cost, steps) = model(x, state, cfg.ACT.HALT_TEST)
                masked_pred = y_pred[1:, :].contiguous().view(-1, 11)
                masked_tgt = y[1:, :].contiguous().view(-1)
                loss = loss_func(masked_pred, masked_tgt) + (cfg.ACT.PENALTY_COEF * torch.mean(ponder_cost))
                all_corrects += norm_corrects(cfg, y_pred.data, y.data, loss_mask).item()
                losses += loss.item()
                ponder_costs += torch.mean(ponder_cost).item()
                acc_steps += torch.mean(steps).item()
        else:
            raise NotImplementedError

    avgs = (elem / (n + 1) for elem in (all_corrects, losses, ponder_costs, acc_steps))
    return avgs
