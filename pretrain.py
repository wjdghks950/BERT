import os
from typing import List, Union
import pickle
import random

import torch
from torch.utils.data import IterableDataset

### You can import any Python standard libraries or pyTorch sub directories here
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader
### END YOUR LIBRARIES

import utils

from bpe import BytePairEncoding
from model import MLMandNSPmodel
from data import ParagraphDataset

# You can use tqdm to check your progress
from tqdm import tqdm, trange

# CUDA_LAUNCH_BLOCKING = 1

class PretrainDataset(IterableDataset):
    def __init__(self, dataset: ParagraphDataset):
        """ Masked Language Modeling & Next Sentence Prediction dataset initializer
        Use below attributes when implementing the dataset

        Attributes:
        dataset -- Paragraph dataset to make a MLM & NSP sample
        """
        self.dataset = dataset

    @property
    def token_num(self):
        return self.dataset.token_num

    def __iter__(self):
        """ Masked Language Modeling & Next Sentence Prediction dataset
        Sample two sentences from the dataset, and make a self-supervised pretraining sample for MLM & NSP

        Note: You can use any sampling method you know.

        Yields:
        source_sentences -- Sampled sentences
        MLM_sentences -- Masked sentences
        MLM_mask -- Masking for MLM
        NSP_label -- NSP label which indicates whether the sentences is connected.

        Example: If 25% mask with 50 % <msk> + 25% random + 25% same -- this percentage is just a example.
        source_sentences = ['<cls>', 'He', 'bought', 'a', 'gallon', 'of', 'milk',
                            '<sep>', 'He', 'drank', 'it', 'all', 'on', 'the', 'spot', '<sep>']
        MLM_sentences = ['<cls>', 'He', '<msk>', 'a', 'gallon', 'of, 'milk',
                         '<sep>', 'He', 'drank', 'it', 'tree', 'on', '<msk>', 'spot', '<sep>']
        MLM_mask = [False, False, True, False, False, False, False,
                    False, True, False, False, True, False True, False, False]
        NSP_label = True
        """
        # Special tokens
        CLS = BytePairEncoding.CLS_token_idx
        SEP = BytePairEncoding.SEP_token_idx
        MSK = BytePairEncoding.MSK_token_idx

        # The number of tokens
        TOKEN_NUM = self.token_num

        while True:
            ### YOUR CODE HERE (~ 40 lines)
            p_change = 0.15
            p_mask = 0.8
            p_random = 0.5
            paragraph_idx = random.sample(list(range(0, len(self.dataset))), 2)

            random_p = random.random()
            if random_p > 0.5: # NSP sample p = 0.5
                p_idx = 0
                while True:
                    p_idx = random.sample(list(range(0, len(self.dataset))), 1)[0]
                    if len(self.dataset[p_idx]) > 1: # Choose paragraph with more than one sentence
                        break
                sent_idx = random.randint(0, len(self.dataset[p_idx]) - 2)
                sentence1 = self.dataset[p_idx][sent_idx]
                sentence2 = self.dataset[p_idx][sent_idx + 1]
                NSP_label = True
            else: # Sample random next sentence from a different paragraph (NSP_label = False)
                random_sent1 = random.randint(0, len(self.dataset[paragraph_idx[0]]) - 1)
                random_sent2 = random.randint(0, len(self.dataset[paragraph_idx[1]]) - 1)
                sentence1 = self.dataset[paragraph_idx[0]][random_sent1]
                sentence2 = self.dataset[paragraph_idx[1]][random_sent2]
                NSP_label = False
            sample_sentence = sentence1 + sentence2
            sent1_len = len(sentence1)
            sent2_len = len(sentence2)
            # Replace with MSK token and replaced words before appending CLS and SEP
            change_idx = random.sample(range(sent1_len + sent2_len), max(1, int((sent1_len + sent2_len) * p_change)))
            mask_idx = random.sample(change_idx, int(len(change_idx) * p_mask))
            random_idx = random.sample(set(change_idx) - set(mask_idx), int((len(change_idx) - len(mask_idx)) * p_random))
            mlm_sentence = sentence1 + sentence2
            MLM_mask = [False] * len(mlm_sentence)
            for idx in change_idx:
                MLM_mask[idx] = True
            for idx in mask_idx:
                mlm_sentence[idx] = MSK
            for w in random_idx:
                while True:
                    random_word = random.randint(5, TOKEN_NUM-1) # TOKEN_NUM == num_vocab
                    if random_word != mlm_sentence[w]:
                        mlm_sentence[w] = random_word
                        break
            # Append CLS and SEP to sentences
            source_sentences = [CLS] + sample_sentence[:sent1_len] + [SEP] + sample_sentence[sent1_len:] + [SEP]
            MLM_sentences = [CLS] + mlm_sentence[:sent1_len] + [SEP] + mlm_sentence[sent1_len:] + [SEP]
            MLM_mask = [False] + MLM_mask[:sent1_len] + [False] + MLM_mask[sent1_len:] + [False]

            ### END YOUR CODE            
            assert len(source_sentences) == len(MLM_sentences) == len(MLM_mask)
            yield source_sentences, MLM_sentences, MLM_mask, NSP_label

def calculate_losses(
    model: MLMandNSPmodel,
    source_sentences: torch.Tensor,
    MLM_sentences: torch.Tensor,
    MLM_mask: torch.Tensor,
    NSP_label: torch.Tensor
):
    """ MLM & NSP losses calculation
    Use cross entropy loss to calculate both MLM and NSP losses.
    MLM loss should be an average loss of masked tokens.

    Arguments:
    model -- MLM & NSP model
    source_sentences -- Source sentences tensor in torch.long type
                        in shape (sequence_length, batch_size)
    MLM_sentences -- Masked sentences tensor in torch.long type
                        in shape (sequence_length, batch_size)
    MLM_mask -- MLM mask tensor in torch.bool type
                        in shape (sequence_length, batch_size)
    NSP_label -- NSP label tensor in torch.bool type
                        in shape (batch_size, )

    Returns:
    MLM_loss -- MLM loss in scala tensor
    NSP_loss -- NSP loss in scala tensor
    """
    ### YOUR CODE HERE (~4 lines)
    mlm_criterion = nn.CrossEntropyLoss()
    nsp_criterion = nn.CrossEntropyLoss()
    out = model(MLM_sentences) # Gives out -> (mlm_logit, nsp_logit)
    target = source_sentences[MLM_mask]
    MLM_loss = mlm_criterion(out[0][MLM_mask], target)
    NSP_loss = nsp_criterion(out[1], NSP_label.long())
    ### END YOUR CODE
    assert MLM_loss.shape == NSP_loss.shape == torch.Size()
    return MLM_loss, NSP_loss

def pretraining(
    model: MLMandNSPmodel,
    model_name: str,
    train_dataset: PretrainDataset,
    val_dataset: PretrainDataset,
):
    """ MLM and NSP pretrainer
    Implement MLM & NSP pretrainer with the given model and datasets.

    Note 1: Don't forget setting model.train() and model.eval() before training / validation.
            It enables / disables the dropout layers of our model.

    Note 2: There are useful tools for your implementation in utils.py

    Note 3: Training takes almost 3 minutes per a epoch on TITAN RTX. Thus, 200 epochs takes 10 hours.
    For those who don't want to wait 10 hours, we attached a model which has trained over 200 epochs.
    You can use it on the IMDB training later.

    Memory tip 1: If you delete the loss tensor explictly after every loss calculation like "del loss",
                  tensors are garbage-collected before next loss calculation so you can cut memory usage.

    Memory tip 2: If you use torch.no_grad when inferencing the model for validation,
                  you can save memory space of gradient.

    Memory tip 3: If you want to keep batch_size while reducing memory usage,
                  creating a virtual batch is a good solution.
    Explanation: https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672

    Useful readings: https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/ 

    Arguments:
    model -- Pretraining model which need to be trained
    model_name -- The model name. You can use this name to save your model
    train_dataset -- Pretraining dataset for training
    val_dataset -- Pretraining dataset for validation

    Variables:
    batch_size -- Batch size
    learning_rate -- Learning rate for the optimizer
    epochs -- The number of epochs
    steps_per_a_epoch -- The number of steps in a epoch.
                        Because there is no end in IterableDataset, you have to set the end of epoch explicitly.
    steps_for_val -- The number of steps for validation

    Returns:
    MLM_train_losses -- List of average MLM training loss per a epoch
    MLM_val_losses -- List of average MLM validation loss per a epoch
    NSP_train_losses -- List of average NSP training loss per a epoch
    NSP_val_losses -- List of average NSP validation loss per a epoch
    """
    # Below options are just our recommendation. You can choose different options if you want.
    batch_size = 16
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 200 # 200 if you want to feel the effect of pretraining
    steps_per_a_epoch: int= 2000
    steps_for_val: int= 200

    ### YOUR CODE HERE
    MLM_train_losses = []
    MLM_val_losses = []
    NSP_train_losses = []
    NSP_val_losses = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device) # Training step
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=utils.pretrain_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=utils.pretrain_collate_fn)
    print('DataLoader Set!')
    print('Batch_size = ', batch_size)
    print('Device: ', device)

    print('============= Training Step =============')
    for epoch in range(epochs):
        model.train()
        trn_combined_loss = trn_mlm_loss = trn_nsp_loss = val_combined_loss = val_mlm_loss = val_nsp_loss = 0.0
        print('[ Epoch: ', epoch + 1, ' ]')
        for steps, (trn_src, trn_mlm, trn_mask, trn_nsp) in enumerate(train_dataloader): # batch => (src, mlm, mask, nsp)
            optimizer.zero_grad()
            if steps >= steps_per_a_epoch:
                break
            trn_src, trn_mlm, trn_mask, trn_nsp = trn_src.to(device), trn_mlm.to(device), trn_mask.to(device), trn_nsp.to(device)
            MLM_loss, NSP_loss = calculate_losses(model, trn_src, trn_mlm, trn_mask, trn_nsp)
            loss_train = MLM_loss + NSP_loss
            loss_train.backward() # retain_graph=True
            optimizer.step()
            trn_combined_loss += loss_train.item()
            trn_mlm_loss += MLM_loss.item()
            trn_nsp_loss += NSP_loss.item()
            if steps % 10 == 0:
                print('[ Epoch/Iter: {0}/{1} | Combined_loss: {2:.5f} | MLM_loss: {3:.5f} | NSP_loss: {4:.5f} ]'.format(epoch+1, steps, loss_train.item(), MLM_loss.item(), NSP_loss.item()))

        avg_trn_loss = trn_combined_loss / steps_per_a_epoch
        avg_trn_mlm_loss = trn_mlm_loss / steps_per_a_epoch
        avg_trn_nsp_loss = trn_nsp_loss / steps_per_a_epoch

        del loss_train, MLM_loss, NSP_loss # Explicitly free the variables (as it won't be garbage collected as long as there is reference to it)

        print('============= Validation Step =============')
        model.eval() # Validation step
        with torch.no_grad():
            for steps, (val_src, val_mlm, val_mask, val_nsp) in enumerate(val_dataloader):
                if steps >= steps_for_val:
                    break
                val_src, val_mlm, val_mask, val_nsp = val_src.to(device), val_mlm.to(device), val_mask.to(device), val_nsp.to(device)
                MLM_loss, NSP_loss = calculate_losses(model, val_src, val_mlm, val_mask, val_nsp)
                loss_val = MLM_loss + NSP_loss
                val_combined_loss += loss_val.item()
                val_mlm_loss += MLM_loss.item()
                val_nsp_loss += NSP_loss.item()
                if steps % 10 == 0:
                    print('[ Epoch/Iter: {0}/{1} | Combined_loss: {2:.5f} | MLM_loss: {3:.5f} | NSP_loss: {4:.5f} ]'.format(epoch+1, steps, loss_val.item(), MLM_loss.item(), NSP_loss.item()))

        avg_val_loss = val_combined_loss / steps_for_val
        avg_val_mlm_loss = val_mlm_loss / steps_for_val
        avg_val_nsp_loss = val_nsp_loss / steps_for_val

        del loss_val, MLM_loss, NSP_loss

        print('[ <Training> Epoch: {0} | Avg. Combined Loss: {1:.5f} | Avg. MLM Loss: {2:.5f} | Avg. NSP Loss {3:.5f} ]'.format(epoch+1, avg_trn_loss, avg_trn_mlm_loss, avg_trn_nsp_loss))
        print('[ <Validation> Epoch: {0} | Avg. Combined Loss: {1:.5f} | Avg. MLM Loss: {2:.5f} | Avg. NSP Loss {3:.5f} ]'.format(epoch+1, avg_val_loss, avg_val_mlm_loss, avg_val_nsp_loss))

        MLM_train_losses.append(avg_trn_mlm_loss)
        NSP_train_losses.append(avg_trn_nsp_loss)
        MLM_val_losses.append(avg_val_mlm_loss)
        NSP_val_losses.append(avg_val_nsp_loss)
    ### END YOUR CODE

    assert len(MLM_train_losses) == len(MLM_val_losses) == epochs and \
           len(NSP_train_losses) == len(NSP_val_losses) == epochs

    assert all(isinstance(loss, float) for loss in MLM_train_losses) and \
           all(isinstance(loss, float) for loss in MLM_val_losses) and \
           all(isinstance(loss, float) for loss in NSP_train_losses) and \
           all(isinstance(loss, float) for loss in NSP_val_losses)

    return MLM_train_losses, MLM_val_losses, NSP_train_losses, NSP_val_losses

##############################################################
# Testing functions below.                                   #
#                                                            #
# We only checks MLM & NSP dataset and loss calculation.     #
# We do not tightly check the correctness of your trainer.   #
# You should attach the loss plot to the report              #
# and submit the pretrained model to validate your trainer.  #
# We will grade the score by running your saved model.       #
##############################################################

def test_MLM_and_NSP_dataset():
    print("======MLM & NSP Dataset Test Case======")
    CLS = BytePairEncoding.CLS_token_idx
    SEP = BytePairEncoding.SEP_token_idx
    MSK = BytePairEncoding.MSK_token_idx

    class Dummy(object):
        def __init__(self):
            self.paragraphs = [[[10] * 100, [11] * 100], [[20] * 100, [21] * 100], [[30] * 100]]

        @property
        def token_num(self):
            return 100

        def __len__(self):
            return len(self.paragraphs)

        def __getitem__(self, index):
            return self.paragraphs[index]
        
    dataset = PretrainDataset(Dummy())

    count = 0
    nsp_true_count = 0
    combinations = set()
    for src, mlm, mask, nsp in dataset:
        # First test
        assert src[0] == mlm[0] == CLS and src[101] == mlm[101] == src[-1] == mlm[101] == SEP and \
               not mask[0] and not mask[101] and not mask[-1], \
                "CLS and SEP should not be masked."
        # Second test
        assert not nsp ^ (src[1] // 10 == src[102] // 10 and src[1] % 10 + 1 == src[102] % 10), \
                "Your result does not match NSP label."
        
        # Third test
        assert all(src[1] == src[i] for i in range(2, 100)) and all(src[102] == src[i] for i in range(103, 201)), \
                "You should not modify the source sentence."

        # Forth test
        assert all((w1 == w2 or m) for w1, w2, m in zip(src, mlm, mask)), \
                "Only masked position can have a different token."

        # Fifth test
        assert .145 < sum(mask) / len(src) < .155, \
                "The number of the masked tokens should be 15%% of the total tokens."

        # Sixth test
        assert .795 < sum(word == MSK for word in mlm) / sum(mask) < .805, \
                "80%% of the masked tokens should be converted to MSK tokens"
        
        # Seventh test
        assert .095 < sum(w1 != w2 for w1, w2 in zip(src, mlm) if w2 != MSK) / sum(mask) < .105, \
            "10%% of the masked tokens should be converted to random tokens"
        
        combinations.add((src[1], src[102]))
        nsp_true_count += nsp
        
        count += 1
        if count > 10000:
            break

    # Eighth test
    assert .45 < nsp_true_count / 10000 < .55, \
            "Your NSP label is biased. Buy a lottery if you failed the test with a correct database."

    # Nineth test
    assert len(combinations) >= 18, \
            "The number of sentence combination is too limited."

    print("MLM & NSP dataset test passed!")

def test_loss_calculation():
    print("======MLM & NSP Loss Calculation Test Case======")
    CLS = BytePairEncoding.CLS_token_idx
    SEP = BytePairEncoding.SEP_token_idx
    MSK = BytePairEncoding.MSK_token_idx
    
    torch.manual_seed(1234)
    model = MLMandNSPmodel(100)

    samples = []

    src = [CLS] + [10, 10, 10, 10, 10, 10] + [SEP] + [20, 20, 20, 20, 20] + [SEP]
    mlm = [CLS] + [10, 10, 10, 10, MSK, 10] + [SEP] + [MSK, 20, 20, 15, 20] + [SEP]
    mask = [False, False, True, False, False, True, False, False, True, False, False, True, False, False]
    nsp = True
    samples.append((src, mlm, mask, nsp))

    src = [CLS] + [30, 30, 30] + [SEP] + [40, 40, 40, 40] + [SEP]
    mlm = [CLS] + [MSK, 30, 30] + [SEP] + [40, 45, 40, 40] + [SEP]
    mask = [False, True, False, True, False, False, True, False, False, False]
    nsp = False
    samples.append((src, mlm, mask, nsp))

    src = [CLS] + [10, 20, 30, 40] + [SEP] + [50, 40, 30, 20, 10] + [SEP]
    mlm = [CLS] + [10, MSK, 30, 40] + [SEP] + [50, MSK, 30, 25, 10] + [SEP]
    mask = [False, False, True, False, False, False, False, True, False, True, False ,False]
    nsp = True
    samples.append((src, mlm, mask, nsp))

    src, mlm, mask, nsp = utils.pretrain_collate_fn(samples)

    MLM_loss, NSP_loss = calculate_losses(model, src, mlm, mask, nsp)

    # First test
    assert MLM_loss.allclose(torch.scalar_tensor(5.12392426), atol=1e-2), \
        "Your MLM loss does not match the expected result"
    print("The first test passed!")

    # Second test
    assert NSP_loss.allclose(torch.scalar_tensor(0.59137219), atol=1e-2), \
        "Your NSP loss does not match the expected result"
    print("The second test passed!")

    print("All 2 tests passed!")

def pretrain_model():
    print("======MLM & NSP Pretraining======")
    """ MLM & NSP Pretraining 
    You can modify this function by yourself.
    This function does not affects your final score.
    """
    train_dataset = ParagraphDataset(os.path.join('data', 'imdb_train.csv'))
    train_dataset = PretrainDataset(train_dataset)
    val_dataset = ParagraphDataset(os.path.join('data', 'imdb_val.csv'))
    val_dataset = PretrainDataset(val_dataset)
    model = MLMandNSPmodel(train_dataset.token_num)

    model_name = 'pretrained'
    print('--------In pretraining(...)--------')
    MLM_train_losses, MLM_val_losses, NSP_train_losses, NSP_val_losses \
            = pretraining(model, model_name, train_dataset, val_dataset)

    torch.save(model.state_dict(), model_name+'_final.pth')

    with open(model_name+'_result.pkl', 'wb') as f:
        pickle.dump((MLM_train_losses, MLM_val_losses, NSP_train_losses, NSP_val_losses), f)

    utils.plot_values(MLM_train_losses, MLM_val_losses, title=model_name + "_mlm")
    utils.plot_values(NSP_train_losses, NSP_val_losses, title=model_name + "_nsp")

    print("Final MLM training loss: {:06.4f}".format(MLM_train_losses[-1]))
    print("Final MLM validation loss: {:06.4f}".format(MLM_val_losses[-1]))
    print("Final NSP training loss: {:06.4f}".format(NSP_train_losses[-1]))
    print("Final NSP validation loss: {:06.4f}".format(NSP_val_losses[-1]))

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(1234)
    torch.manual_seed(1234)

    test_MLM_and_NSP_dataset()
    test_loss_calculation()
    pretrain_model()
