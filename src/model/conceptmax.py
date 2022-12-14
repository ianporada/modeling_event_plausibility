from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class ConceptMax(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        lengths: Optional[List[int]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # feed all abstractions through the transformer together
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        # group logits by example
        grouped_logits = torch.split(logits, lengths, dim=0)
        
        # reduce logits using 'logsumexp' for training or 'max' for testing
        reduced_logits = None
        if self.training:
            reduced_logits = [torch.logsumexp(item_logits, dim=0)
                              for item_logits in grouped_logits]
        else:
            reduced_logits = [torch.max(item_logits, dim=0).values
                              for item_logits in grouped_logits]
        logits = torch.concat(reduced_logits)
        
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
