import os
import pdb
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        # Only greedy decoding is supported now
        return self.MaxDecode(nn_output, vid_lgt)

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            valid_len = int(vid_lgt[batch_idx].item())  # Convert tensor to int
            group_result = [x[0] for x in groupby(index_list[batch_idx][:valid_len])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                            enumerate(max_result)])
        return ret_list
    
    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
        return ret_list

    # def MaxDecode(self, nn_output, vid_lgt):
    #     index_list = torch.argmax(nn_output, axis=2)
    #     batchsize, lgt = index_list.shape
    #     ret_list = []
    #     for batch_idx in range(batchsize):
    #         group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
    #         filtered = [*filter(lambda x: x != self.blank_id, group_result)]
    #         if len(filtered) > 0:
    #             max_result = torch.stack(filtered)
    #             max_result = [x[0] for x in groupby(max_result)]
    #         else:
    #             max_result = filtered
    #         ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
    #                          enumerate(max_result)])
    #     return ret_list
