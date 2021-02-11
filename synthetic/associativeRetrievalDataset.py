import numpy as np
import torch


def int_to_lex(integer, n=26):
    """ Convert integer to string with alphabet of size n"""
    assert 0 < n < 27, "n has to 0 < n < 27"
    assert type(integer) is int, "arg has to be positive"

    if integer == 0:
        return ""

    vocab = [chr(97+i) for i in range(n)]
    z = integer

    length = np.int(np.log(z) / np.log(n)) + 1
    lex = ""
    for i in reversed(range(length)):
        count = np.int(z / n**i)  # can fit how many n^i?
        z = z - count * n ** i
        lex += vocab[count-1]
    return lex


def lex_to_int(lex, n=26):
    """ Convert string with alphabet of size n to integer """
    assert 0 < n < 27, "n has to 0 < n < 27"
    assert type(lex) is str, "arg has to be str"
    lex = lex.lower()
    assert all(97 <= ord(c) < 97 + n for c in list(lex)), "has invalid characters"
    counts = [ord(c)-97+1 for c in list(lex)]
    power = len(counts) - 1
    total = 0
    for c in counts:
        total += c * n**power
        power -= 1
    return total


class DataLoader:
    def __init__(self, seq_len, n_values, n_keys, replace):
        self.seq_len = seq_len
        self.n_values = n_values
        self.n_keys = n_keys
        self.replace = replace

        if not replace:
            assert self.n_keys >= self.seq_len, \
                "to be unambiguous without replacement, " \
                + "we cannot have fewer keys than sequence elements"

        self._create_vocab()

    def _create_vocab(self):
        self.itos = []  # integer to string
        self.key_idxs = []
        for i in range(1, self.n_keys+1):
            self.itos.append(int_to_lex(i).lower())
            self.key_idxs.append(len(self.itos) - 1)

        self.value_idxs = []
        for i in range(1, self.n_values+1):
            self.itos.append(int_to_lex(i).upper())
            self.value_idxs.append(len(self.itos) - 1)

        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def get_name(self):
        return f"AssocSeq{self.seq_len}N{self.n_keys}R{1 if self.replace else 0}"

    def get_batch(self, batch_size, device):
        """ Returns random batch of input sequences each with one query. """
        batch_x, batch_q, batch_y = [], [], []
        for _ in range(batch_size):
            k = np.random.choice(self.key_idxs,
                                 size=(self.seq_len,),
                                 replace=self.replace)
            
            # need to also sample multiple value if seq_len is longer
            v = np.random.choice(self.value_idxs,
                                 size=(self.seq_len,),
                                 replace=self.replace)  
            
            query_idx = np.where(k == np.random.choice(k))[0][-1]  # last index
            q = k[query_idx]
            y = v[query_idx]
            
            x = np.stack([k, v], axis=0)
            batch_x.append(x)
            batch_q.append([q, ])
            batch_y.append([y, ])
        return (torch.tensor(batch_x).to(device), 
                torch.tensor(batch_q).to(device), 
                torch.tensor(batch_y).to(device))
    
    def get_all_queries(self, device, max_queries=200):
        """ Return batch with the same input sequence but different queries. """
        batch_x, batch_q, batch_y = [], [], []
        # pick one sequence
        k = np.random.choice(self.key_idxs,
                             size=(len(self.key_idxs),),
                             replace=self.replace)

        v = np.random.choice(self.value_idxs,
                             size=(len(self.key_idxs),),
                             replace=self.replace)
        
        # check all associations
        possible_idxs = [np.where(k == possible_k)[0][-1] 
                         for possible_k in np.unique(k)][:max_queries]
        for query_idx in possible_idxs:
            q = k[query_idx]
            y = v[query_idx]
            
            x = np.stack([k, v], axis=0)
            batch_x.append(x)
            batch_q.append([q, ])
            batch_y.append([y, ])
        
        return (torch.tensor(batch_x).to(device), 
                torch.tensor(batch_q).to(device), 
                torch.tensor(batch_y).to(device))

    def batch_to_string(self, x, q, y):
        """ Converts a batch into a string representation. """
        txt = ""
        bs = x.shape[0]
        for i in range(bs):
            sample_txt = ""

            # append the input sequence of associations
            seq_lst = []
            for seq_el in x[i].unbind(-1):
                a, b = seq_el.unbind()
                seq_lst.append("{}->{}".format(self.itos[a.item()], self.itos[b.item()]))
            sample_txt += ", ".join(seq_lst)

            # append query q
            sample_txt += "\t{}?".format(self.itos[q[i].item()])

            # append the correct answer/target
            sample_txt += "\t{}".format(self.itos[y[i].item()])

            txt += sample_txt
            txt += "\n"
        return txt

    def __repr__(self):
        txt = f'''Associative Retrieval DataLoader
\tsequence length: {self.seq_len}
\tnumber of keys: {self.n_keys}
\tnumber of values: {self.n_values}
\tvocabulary size: {len(self.itos)}
\tdraw sample with replacement: {self.replace}'''
        return txt


if __name__ == "__main__":
    print("Example with ...")
    n_keys = 10
    n_values = 10
    print("{} keys (lower case characters)".format(n_keys))
    print("{} values (upper case characters)".format(n_values))

    # without replacement
    print("\n\n# without replacement - unambiguous associative retrieval")
    seq_len = 5
    batch_size = 10
    print("batch_size: ", batch_size)
    print("sequence_length: ", seq_len)
    dataloader = DataLoader(seq_len=seq_len,
                            n_values=n_values,
                            n_keys=n_keys,
                            replace=False)
    itos = dataloader.itos
    print("vocab: ", itos)
    print("keys: ", [itos[i] for i in dataloader.key_idxs])
    print("values: ", [itos[i] for i in dataloader.value_idxs])

    print("\n## get a random batch with randomly chosen queries")
    print("x, q, y = dataloader.get_batch(batch_size=batch_size)")
    x, q, y = dataloader.get_batch(batch_size=batch_size, device="cpu")
    print("x.shape: ", x.shape)
    print(x)
    print("q.shape: ", q.shape)
    print(q)
    print("y.shape: ", y.shape)
    print(y)
    print("text representation:")
    print(dataloader.batch_to_string(x, q, y))

    print("\n## get a random sample with all possible queries")
    print("x, q, y = dataloader.get_all_queries(batch_size=batch_size)")
    x, q, y = dataloader.get_all_queries(device="cpu")
    print("x.shape: ", x.shape)
    print(x)
    print("q.shape: ", q.shape)
    print(q)
    print("y.shape: ", y.shape)
    print(y)
    print("text representation:")
    print(dataloader.batch_to_string(x, q, y))

    # with replacement
    print("\n\n# with replacement - associative retrieval with updates (keep last one)")
    seq_len = 5
    batch_size = 4
    print("batch_size: ", batch_size)
    print("sequence_length: ", seq_len)
    dataloader = DataLoader(seq_len=seq_len,
                            n_values=n_values,
                            n_keys=n_keys,
                            replace=True)
    itos = dataloader.itos

    print("\n## get a random batch with randomly chosen queries")
    print("x, q, y = dataloader.get_batch(batch_size=batch_size)")
    x, q, y = dataloader.get_batch(batch_size=batch_size, device="cpu")
    print("x.shape: ", x.shape)
    print(x)
    print("q.shape: ", q.shape)
    print(q)
    print("y.shape: ", y.shape)
    print(y)
    print("text representation:")
    print(dataloader.batch_to_string(x, q, y))

    print("\n## get a random sample with all possible queries")
    print("x, q, y = dataloader.get_all_queries(batch_size=batch_size)")
    x, q, y = dataloader.get_all_queries(device="cpu")
    print("x.shape: ", x.shape)
    print(x)
    print("q.shape: ", q.shape)
    print(q)
    print("y.shape: ", y.shape)
    print(y)
    print("text representation:")
    print(dataloader.batch_to_string(x, q, y))
