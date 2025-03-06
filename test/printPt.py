import torch


if __name__ == '__main__':

    data = torch.load('./best.pt')

    if isinstance(data, dict):
        print("\nKeys in the dictionary:")
        for key in data.keys():
            print(key)
            print(data[key].shape if hasattr(data[key], 'shape') else data[key])
    elif isinstance(data, torch.nn.Module):
        print("\nModel structure:")
        print(data)
    else:
        print("\nUnknown data type.")