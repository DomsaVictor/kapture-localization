# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from abc import ABC, abstractmethod
import torch
import numpy as np
from kapture_localization.utils.logging import getLogger


class MatchPairGenerator(ABC):
    @abstractmethod
    def match_descriptors(self, descriptors_1, descriptors_2):
        raise NotImplementedError()


class MatchPairNnTorch(MatchPairGenerator):
    def __init__(self, use_cuda=True, matcher="L2"):
        super().__init__()
        self._device = torch.device("cuda:0"
                                    if use_cuda and torch.cuda.is_available()
                                    else "cpu")
        self.min_cossim = 0.82
        if matcher == "L2":
            self.match_descriptors = self.match_descriptors_L2
        elif "mnn" in matcher:
            if matcher.split("_") == 2:
                self.min_cossim = float(matcher.split("_")[1])
            self.match_descriptors = self.match_descriptors_mnn


    def match_descriptors(self, descriptors_1, descriptors_2):
        raise NotImplementedError()


    def match_descriptors_L2(self, descriptors_1, descriptors_2):
        if descriptors_1.shape[0] == 0 or descriptors_2.shape[0] == 0:
            return np.zeros((0, 3))

        # send data to GPU
        descriptors1_torch = torch.from_numpy(descriptors_1).to(self._device)
        descriptors2_torch = torch.from_numpy(descriptors_2).to(self._device)
        # make sure its double (because CUDA tensors only supports floating-point)
        descriptors1_torch = descriptors1_torch.float()
        descriptors2_torch = descriptors2_torch.float()
        # sanity check
        if not descriptors1_torch.device == self._device:
            getLogger().debug('descriptor on device {} (requested {})'.format(descriptors1_torch.device, self._device))
        if not descriptors2_torch.device == self._device:
            getLogger().debug('descriptor on device {} (requested {})'.format(descriptors2_torch.device, self._device))

        simmilarity_matrix = descriptors1_torch @ descriptors2_torch.t()
        scores = torch.max(simmilarity_matrix, dim=1)[0]
        nearest_neighbor_idx_1vs2 = torch.max(simmilarity_matrix, dim=1)[1]
        nearest_neighbor_idx_2vs1 = torch.max(simmilarity_matrix, dim=0)[1]
        ids1 = torch.arange(0, simmilarity_matrix.shape[0], device=descriptors1_torch.device)
        # cross check
        mask = ids1 == nearest_neighbor_idx_2vs1[nearest_neighbor_idx_1vs2]
        matches_torch = torch.stack(
            [ids1[mask].type(torch.float), nearest_neighbor_idx_1vs2[mask].type(torch.float), scores[mask]]).t()
        # retrieve data back from GPU
        matches = matches_torch.data.cpu().numpy()
        matches = matches.astype(np.float)
        return matches

    
    def match_descriptors_mnn(self, descriptors_1, descriptors_2):

        # send data to GPU
        descriptors1_torch = torch.from_numpy(descriptors_1).to(self._device)
        descriptors2_torch = torch.from_numpy(descriptors_2).to(self._device)
        # make sure its double (because CUDA tensors only supports floating-point)
        descriptors1_torch = descriptors1_torch.float()
        descriptors2_torch = descriptors2_torch.float()
        # sanity check
        if not descriptors1_torch.device == self._device:
            getLogger().debug('descriptor on device {} (requested {})'.format(descriptors1_torch.device, self._device))
        if not descriptors2_torch.device == self._device:
            getLogger().debug('descriptor on device {} (requested {})'.format(descriptors2_torch.device, self._device))


        cossim = descriptors1_torch @ descriptors2_torch.t()
        cossim_t = descriptors2_torch @ descriptors1_torch.t()
        
        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0
        
        if self.min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > self.min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1