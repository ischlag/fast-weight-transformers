import torch
import csv
import os

NAN_CHECK_ON = False


def check(tensor, shape):
    """ Checks the shape of the tensor for better code redability and bug prevention. """
    assert isinstance(tensor, torch.Tensor), "SHAPE GUARD: tensor is not torch.Tensor!"
    tensor_shape = list(tensor.shape)
    assert len(shape) == len(tensor_shape), f"SHAPE GUARD: tensor shape {tensor_shape} not the same length as {shape}"
    
    for idx, (a, b) in enumerate(zip(tensor_shape, shape)):
        if b <= 0:
            continue  # ignore -1 sizes
        else:
            assert a == b, f"SHAPE GUARD: at pos {str(idx)}, tensor shape {tensor_shape} does not match {shape}"


def nan_checker(x):
    if NAN_CHECK_ON:
        assert torch.isnan(x).sum().item() == 0, f"NaN: {x}"
        assert torch.isinf(x).sum().item() == 0, f"INF: {x}"


class CsvWriter:
  def __init__(self, column_names, path, file_name):
    self.csv_file = os.path.join(path, file_name)
    self.file = open(self.csv_file, "w+")
    self.writer = csv.writer(self.file)
    self.writer.writerow(column_names)

  def write(self, values):
    self.writer.writerow(values)
    self.file.flush()

  def close(self):
    self.file.close()
