import itertools

import torch
from torch.utils.data import IterableDataset, get_worker_info


class PreprocessedIterableDataset(IterableDataset):
    
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        # batch = []
        residule = []
        
        for example in iter_data:
            tokenized_example = self.tokenizer(example["text"])["input_ids"] + [self.tokenizer.eos_token_id] # tokenize后首位会被加上bos
            assert type(tokenized_example) == list, type(tokenized_example)
            assert tokenized_example[0] == self.tokenizer.bos_token_id, f"{tokenized_example[0]}"
            assert tokenized_example[-1] == self.tokenizer.eos_token_id, f"{tokenized_example[-1]}"
            residule += tokenized_example
            
            while(len(residule) >= self.max_length):
                yield residule[:self.max_length]
                residule = residule[self.max_length:]
                # if len(batch) == self.batch_size:
                #     yield {"input_ids": batch}
                #     batch = []