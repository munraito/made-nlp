import random
import imp

import torch
import tqdm
from nltk.translate.bleu_score import corpus_bleu

import utils


imp.reload(utils)
translate_sentence_vectorized = utils.translate_sentence_vectorized
get_text = utils.get_text
count_parameters = utils.count_parameters


def bleu_score(model, test_iterator, TRG, transformer=False): # , bert=False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_text = []
    generated_text = []
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_iterator)):
            src = batch.src
            trg = batch.trg
            if transformer:
                translation, _ = translate_sentence_vectorized(src, TRG, model, device)
                generated_text.extend(translation)
                original_text.extend([get_text(x, TRG.vocab) for x in trg])
            else:
                output = model(src, trg, 0) #turn off teacher forcing
#                 if bert:
#                     output = output.argmax(dim=-1).cpu().numpy()
#                     generated_text.extend([get_text(x, TRG.vocab) for x in list(output[1:, 0])])
#                     original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy().T])            
#                 else:    
                output = output.argmax(dim=-1)
                generated_text.extend([get_text(x, TRG.vocab) for x in output[1:].detach().cpu().numpy().T])
                original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy().T])            
            
    score = corpus_bleu([[text] for text in original_text], generated_text) * 100
    return score, original_text, generated_text


def show_results(model, test_iterator, TRG, tr_flag=False, num_examples=5): # bert_flag=False, 
    print(f'The model has {count_parameters(model):,} trainable parameters')
    score, original_text, generated_text = bleu_score(model, test_iterator, TRG, transformer=tr_flag) # , bert=bert_flag
    print('BLEU:', score)
    print()
    for _ in range(num_examples):
        index = random.randint(0, len(original_text))
        print('original:', ' '.join(original_text[index]))
        print('translated:', ' '.join(generated_text[index]))
        print()
