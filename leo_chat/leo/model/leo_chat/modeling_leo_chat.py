# --------------------------------------------------------
# LEO
# Copyright (c) 2024 Waterloo's Wiselab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from leo.conversation import get_conv_template
from leo.model.llm.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from leo.model.llm.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .configuration_leo_chat import LeoConfig
from leo.model.projector.modeling_projector import MLP
from leo.model.vision_encoder.intern_vit.modeling_intern_vit import InternVisionModel
from leo.model.leo_chat.modeling_internvl_chat import InternVLChatModel
from leo.model.vision_encoder.sam_vit.sam_encoder import SAMModel
from leo.model.vision_encoder.sam_vit.configuring_sam import SamVitConfig

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class LeoModel(PreTrainedModel):
    config_class = LeoConfig
    # _no_split_modules = []
    _no_split_modules = ['InternVisionModel', 'SAMModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: LeoConfig, vision_model_1=None, mlp1=None, language_model=None, vision_model_2=None):
        super().__init__(config)
  
        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config_1.image_size
        patch_size = config.vision_config_1.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2)) * 2
        self.downsample_ratio = config.downsample_ratio
        self.downsample_ratio2 = config.downsample_ratio2
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        vit_hidden_size = config.vision_config_1.hidden_size # 1024
        self.llm_hidden_size = config.llm_config.hidden_size
        
        if vision_model_1 is not None:
            self.vision_model_1 = vision_model_1
        else:
            self.vision_model_1 = InternVisionModel(config.vision_config_1)
            
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')
                
        if mlp1 is not None:
            self.mlp1 = mlp1
        else:
            self.mlp1 = MLP(vit_hidden_size, self.downsample_ratio, self.llm_hidden_size)
            
        if vision_model_2 is not None:
            self.vision_model_2 = vision_model_2
        else:
            self.vision_model_2 = SAMModel(config.vision_config_2).vision_encoder
        
        self.mlp2 = MLP(vit_hidden_size, self.downsample_ratio, self.llm_hidden_size)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0


    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1',
                              'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
        
    
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            pixel_values_sam: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        
        vit_embeds_1 = self.extract_feature_vision_1(pixel_values)
        vit_embeds_1 = vit_embeds_1[image_flags == 1]
        
        vit_embeds_2 = self.extract_feature_vision_2(pixel_values_sam)
        vit_embeds_2 = vit_embeds_2[image_flags == 1]
        
        vit_batch_size = pixel_values.shape[0]
        if vit_embeds_1.shape[0] == 0:
            #handling pure text scenarios
            merged_vit_embeds = torch.empty(0, 1, self.llm_hidden_size, 
                                            dtype=torch.bfloat16, device=pixel_values.device)
        else:
            V_b, V_n, V_c = vit_embeds_1.shape
            merged_vit_embeds = torch.empty(V_b, 2 * V_n, V_c, dtype=vit_embeds_1.dtype, device=vit_embeds_1.device)
            merged_vit_embeds[:, 0::2] = vit_embeds_1  # Place vit_embeds_1 in even indices
            merged_vit_embeds[:, 1::2] = vit_embeds_2 

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 +  merged_vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            merged_vit_embeds =  merged_vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={ merged_vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 +  merged_vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_unshuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size() 
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor)) 
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous() 
        
        return x

    def extract_feature_vision_1(self, pixel_values):
        if self.select_layer == -1: 
            vit_embeds = self.vision_model_1(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state 
        else:
            vit_embeds = self.vision_model_1(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer] 
        vit_embeds = vit_embeds[:, 1:, :] 

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1) 
        vit_embeds = self.pixel_unshuffle(vit_embeds, scale_factor=self.downsample_ratio) 
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        
        return vit_embeds

    def pixel_unshuffle_sam(self, x, scale_factor=4):
        x = nn.functional.pixel_unshuffle(x, int(scale_factor)) # N , C , H, W
        x = x.permute(0, 2, 3, 1).contiguous() # --> N, H, W, C
        
        return x
        

    def extract_feature_vision_2(self, pixel_values_sam):
        vit_embeds_sam = self.vision_model_2(pixel_values_sam,
                                             output_hidden_states=False,
                                             return_dict=True).last_hidden_state

        vit_embeds_sam = self.pixel_unshuffle_sam(vit_embeds_sam, scale_factor=self.downsample_ratio2) 
        vit_embeds_sam = vit_embeds_sam.reshape(vit_embeds_sam.shape[0], -1, vit_embeds_sam.shape[-1]) 
        vit_embeds_sam = self.mlp2(vit_embeds_sam)

        return vit_embeds_sam

    def chat(self,
             tokenizer,
             pixel_values,
             pixel_values_sam,
             question,
             generation_config,
             history=None,
             return_history=False,
             num_patches_list=None,
             IMG_START_TOKEN='<img>',
             IMG_END_TOKEN='</img>',
             IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            pixel_values_sam=pixel_values_sam,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    def batch_chat(self,
                   tokenizer,
                   pixel_values,
                   pixel_values_sam,
                   questions,
                   generation_config,
                   num_patches_list=None,
                   history=None,
                   return_history=False,
                   IMG_START_TOKEN='<img>',
                   IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                   verbose=False, image_counts=None):
        
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            pixel_values_sam=pixel_values_sam,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_values_sam: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds_1 = self.extract_feature_vision_1(pixel_values)
                vit_embeds_2 = self.extract_feature_vision_2(pixel_values_sam)

            V_b, V_n, V_c = vit_embeds_1.shape
            merged_vit_embeds = torch.empty(V_b, 2 * V_n, V_c, dtype=vit_embeds_1.dtype, device=vit_embeds_1.device)
            merged_vit_embeds[:, 0::2] = vit_embeds_1  # Place vit_embeds_1 in even indices
            merged_vit_embeds[:, 1::2] = vit_embeds_2 
         
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = merged_vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs