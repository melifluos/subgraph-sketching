import os, json

def get_text_graph(dataset: str, 
                   use_text: bool = False, 
                   use_gpt : bool =False, 
                   seed: int =0):
    if dataset == 'cora':
        from load_cora import get_raw_text_cora as get_raw_text
    elif dataset == 'pubmed':
        from load_pubmed import get_raw_text_pubmed as get_raw_text
    elif dataset == 'ogbn-arxiv':
        from load_arxiv import get_raw_text_arxiv as get_raw_text
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(use_text=False, seed=seed)
        return data
    else:# for finetuning LM
        data, text = get_raw_text(use_text=True, seed=seed)
        return data, text

if __name__ == '__main__':
    data, text = get_text_graph('cora', use_text=True) 
    print(data)
    print(text)