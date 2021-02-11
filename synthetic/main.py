import numpy as np
import argparse
import torch
import time
import lib

from linearAttention import PrefixSumLinearAttentionModel
from softmaxAttention import SoftmaxAttentionModel
from associativeRetrievalDataset import DataLoader
from lib import check, nan_checker
from lib import CsvWriter

lib.NAN_CHECK_ON = False
np.set_printoptions(suppress=False, edgeitems=100)


def train(dataloader, model, steps, lr, device, batch_size, log_every,
          test_every, test_sequences, stop_criterion, log_folder):
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    sum_duration = 0
    count = 0
    best_loss = np.inf
    strike = 0
    max_strike = 10
    file_name = model.get_name() + "_" + dataloader.get_name() + ".csv"
    csv_writer = CsvWriter(column_names=["step", "eval-loss"],
                           path=log_folder,
                           file_name=file_name)
    print("Logging to ... ", csv_writer.csv_file)
    seq_len = dataloader.seq_len
    
    for step in range(1, steps+1):
        model.train()

        # get batch
        batch_x, batch_q, batch_y = dataloader.get_batch(batch_size, device)
        check(batch_x, [batch_size, 2, seq_len])

        # forward pass
        start_time = time.time()
        batch_y_hat = model(batch_x, batch_q)
        check(batch_y_hat, [batch_size, model.n_values])
        check(batch_y, [batch_size, 1])

        # get target vectors
        with torch.no_grad():
            # due to the somewhat awkward contraint of not training the value implementations the values
            # are have their own not trained embedding which requires us to properly offset the indecies.
            batch_y = model.embeddingV(batch_y - model.n_values).squeeze(1)
        check(batch_y, [batch_size, model.n_values])

        # reconstruction loss
        loss = 0.5 * (batch_y - batch_y_hat).pow(2)
        loss = loss.sum(dim=-1).mean()
        nan_checker(loss)

        # gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_duration += time.time() - start_time
        count += 1

        # terminal log
        if step % log_every == 0 and step != 0:
            print("step {:5}: loss={:.4f}".format(step, loss))

        # evaluation
        if step % test_every == 0 and step != 0:
            losses = []
            test_seconds = []
            test_count = 0
            for _ in range(test_sequences):
                model.eval()

                # get eval batches
                full_batch_x, full_batch_q, full_batch_y = dataloader.get_all_queries(device)
                # here batch size is in fact the number of keys due to the get_all_queries call

                # split large eval batch into batches no larger than training batch_size
                full_bs = full_batch_x.shape[0]
                n_splits = full_bs // batch_size
                rest = full_bs % batch_size
                if rest > 0:
                    splits = [batch_size] * n_splits + [rest]
                else:
                    splits = [batch_size] * n_splits

                batches_x = torch.split(full_batch_x, splits, dim=0)
                batches_q = torch.split(full_batch_q, splits, dim=0)
                batches_y = torch.split(full_batch_y, splits, dim=0)

                for i in range(len(splits)):
                    batch_x = batches_x[i]
                    batch_q = batches_q[i]
                    batch_y = batches_y[i]
                    bs = batch_x.shape[0]

                    # eval forward pass
                    test_start_time = time.time()
                    batch_y_hat = model(batch_x, batch_q)
                    test_seconds.append(time.time() - test_start_time)
                    test_count += 1
                    check(batch_y_hat, [bs, model.n_values])
                    check(batch_y, [bs, 1])

                    batch_y = model.embeddingV(batch_y - model.n_values).squeeze(1)
                    check(batch_y, [bs, model.n_values])
                    loss = 0.5 * (batch_y - batch_y_hat).pow(2)
                    loss = loss.sum(dim=-1).mean()

                    losses.append(loss.cpu().detach().numpy())

            loss_mean = np.mean(losses)
            if loss_mean < best_loss:
                best_loss = loss_mean
                strike = 0
            else:
                strike = strike + 1

            print("train batches/s={:.1f}  test batches/s={:.1f}"
                  .format(count/sum_duration, test_count/np.sum(test_seconds)))
            if device == "cuda":
                print("peak memory allocation={:.1f} * 1000^2 bytes (megabytes)"
                      .format(torch.cuda.max_memory_allocated(0) / 1000**2))  # megabytes (mega = 1000**2)
            print("test loss={:.4f} \n".format(loss_mean))
            sum_duration = count = 0
            csv_writer.write((step, loss_mean))

            # stop training if criterion is met or there was no progress
            if loss_mean <= stop_criterion or strike > max_strike:
                break


def run(n_keys, seq_len, hidden_size, attn_name, batch_size, attn_arg, replace, update_rule, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data generator parameters
    n_values = n_keys

    # model parameters
    embedding_size = int((n_values + n_keys) * 1.1)

    # optimisation parameters
    max_steps = 100000  # 100k
    log_every = 25
    test_every = 100
    test_sequences = 20
    stop_criterion = 0.001
    log_folder = "logs"
    learning_rate = 1e-3  # Adam default

    dataloader = DataLoader(seq_len=seq_len, n_values=n_values, n_keys=n_keys, replace=replace)

    if attn_name == "softmax":
        model = SoftmaxAttentionModel(embedding_size=embedding_size,
                                      hidden_size=hidden_size,
                                      n_values=n_values,
                                      n_keys=n_keys)
    else:
        model = PrefixSumLinearAttentionModel(embedding_size=embedding_size,
                                              hidden_size=hidden_size,
                                              n_values=n_values,
                                              n_keys=n_keys,
                                              attention_type=attn_name,
                                              update_rule=update_rule,
                                              arg=attn_arg)

    print("device: ", device)
    print(dataloader)
    print(model)
    if type(model) is PrefixSumLinearAttentionModel:
        print(model.string_repr())

    print("parameters (trainable):")
    for idx, p in enumerate(model.parameters()):
        print("[{}]: {} ({})".format(idx, list(p.shape), p.requires_grad))

    if update_rule == "fwm" and attn_name == "tanh":
        # necessary for FWM
        learning_rate = 1e-4
        batch_size = 128

    train(dataloader=dataloader,
          model=model,
          steps=max_steps,
          lr=learning_rate,
          device=device,
          batch_size=batch_size,
          log_every=log_every,
          test_every=test_every,
          test_sequences=test_sequences,
          stop_criterion=stop_criterion,
          log_folder=log_folder)
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train one model with various numer of unique keys and values.")
    parser.add_argument("--begin", type=int, help="number of unique keys S; size of smallest experiment", default=20)
    parser.add_argument("--end", type=int, help="number of unique keys S; size of largest experiment", default=20)
    parser.add_argument("--step", type=int, help="increase of S until largest experiment is reached", default=20)
    parser.add_argument("--hidden_size", type=int, help="size of the keys before fed into the phi function", default=64)
    parser.add_argument("--seed", type=int, help="seed", default=1)
    parser.add_argument("--batch_size", type=int, help="batchsize for train and test", default=32)
    parser.add_argument("--attn_name", default="softmax",
                        help="which attention type is used (softmax, linear, favor, dpfp")
    parser.add_argument("--attn_arg", type=int, help="int argument of the respective attention type", default=0)
    parser.add_argument("--replace", dest="replace", action="store_true", default=False,
                        help="sample keys with replacement")
    parser.add_argument("--update_rule", help="name of the update rule (sum, fwm, ours)", default="sum")
    args = parser.parse_args()

    print(args)

    for N in range(args.begin, args.end+1, args.step):
        run(n_keys=N,
            seq_len=N*2 if args.replace else N,
            hidden_size=args.hidden_size,
            attn_name=args.attn_name,
            batch_size=args.batch_size,
            attn_arg=args.attn_arg,
            replace=args.replace,
            update_rule=args.update_rule)
        torch.cuda.empty_cache()
