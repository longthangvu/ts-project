import os
import torch
# from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    # Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    # Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    # WPMixer, MultiPatchFormer
from models import TimesNet, Nonstationary_Transformer, FEDformer, PatchTST, iTransformer, PAttn, TSMixer, TimeMixer

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'FEDformer': FEDformer,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'PAttn': PAttn,
            'TSMixer': TSMixer,
            'TimeMixer': TimeMixer,
            # 'TimeXer': TimeXer
            # 'DLinear': DLinear,
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Informer': Informer,
            # 'LightTS': LightTS,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            # 'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            # 'Crossformer': Crossformer,
            # 'FiLM': FiLM,
            # 'Koopa': Koopa,
            # 'TiDE': TiDE,
            # 'FreTS': FreTS,
            # 'MambaSimple': MambaSimple,
            # 'TSMixer': TSMixer,
            # 'SegRNN': SegRNN,
            # 'TemporalFusionTransformer': TemporalFusionTransformer,
            # "SCINet": SCINet,
            # 'PAttn': PAttn,
            # 'WPMixer': WPMixer,
            # 'MultiPatchFormer': MultiPatchFormer
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model()
        if self.__class__.__name__ != 'Exp_Tradition_Longterm':
            self.model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
