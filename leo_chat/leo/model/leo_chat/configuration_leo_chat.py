# --------------------------------------------------------
# LEO
# Copyright (c) 2024 Waterloo's Wiselab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from leo.model.llm.internlm2.configuration_internlm2 import InternLM2Config
from leo.model.llm.phi3.configuration_phi3 import Phi3Config
from leo.model.vision_encoder.intern_vit.configuration_intern_vit import InternVisionConfig
from leo.model.vision_encoder.sam_vit.configuring_sam import SamVitConfig
from leo.model.leo_chat.configuration_internvl_chat import InternVLChatConfig
from transformers import AutoConfig, LlamaConfig, Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class LeoConfig(PretrainedConfig):
    model_type = 'hybrid_chat'
    is_composition = True

    def __init__(
            self,
            vision_config_1=None,
            vision_config_2=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            downsample_ratio=0.5,
            downsample_ratio2=4,
            pad2square=False,
            template=None,
            select_layer=-1,
            force_image_size=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v2',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            **kwargs):

        super().__init__(**kwargs)

        if vision_config_1 is None:
            vision_config = {}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if vision_config_2 is None:
            vision_config_2 = SamVitConfig()
            logger.info('sam_config is None. Initializing the samVitConfig with default values.')

        if llm_config is None:
            llm_config = {}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')
        
        
        self.vision_config_1 = InternVisionConfig(**vision_config_1)
        self.vision_config_2 = SamVitConfig(**(vision_config_2 if isinstance(vision_config_2, dict) else vision_config_2.to_dict()))
        if llm_config['architectures'][0] == 'InternLM2ForCausalLM':
            self.llm_config = InternLM2Config(**llm_config)
        elif llm_config['architectures'][0] == 'Phi3ForCausalLM':
            self.llm_config = Phi3Config(**llm_config)
        else:
            raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.downsample_ratio2 = downsample_ratio2
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.force_image_size = force_image_size

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config_1'] = self.vision_config_1.to_dict()
        output['vision_config_2'] = self.vision_config_2.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['pad2square'] = self.pad2square
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['downsample_ratio2'] = self.downsample_ratio2
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        return output
