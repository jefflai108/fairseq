# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging.meters import safe_round


@register_criterion("wav2vec_kd")
class Wav2vecCriterion(FairseqCriterion):
    def __init__(self, task, 
                 infonce=False, 
                 loss_weights=None, 
                 log_keys=None, 
                 distill_with_kl=False, 
                 distill_with_ce=False,
                 distill_with_ce_mix=False):
        super().__init__(task)
        self.infonce = infonce
        self.loss_weights = None if loss_weights is None else eval(loss_weights)
        self.log_keys = [] if log_keys is None else eval(log_keys)
        self.teacher_model = None
        self.distill_with_kl = distill_with_kl
        self.distill_with_ce = distill_with_ce
        self.distill_with_ce_mix = distill_with_ce_mix

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--infonce', action='store_true',
                            help='if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)')
        parser.add_argument('--loss-weights', type=str, default=None,
                            help='weights for additional loss terms (not first one)')
        parser.add_argument('--log-keys', type=str, default=None,
                            help='output keys to log')
        parser.add_argument('--distill-with-kl', action='store_true',
                            help='KD with KL loss')
        parser.add_argument('--distill-with-ce', action='store_true',
                            help='KD with CE loss from teacher')
        parser.add_argument('--distill-with-ce-mix', action='store_true',
                            help='KD with CE loss from teacher + data')
        # fmt: on

    def add_teacher(self, teacher_model):
        """
        add teacher model for KD
        """
        self.teacher_model = teacher_model
        self.teacher_model.eval()

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # student
        teacher_sample = copy.deepcopy(sample)
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        mod_logits = model.get_mod_logits(net_output).float()
        neg_is_pos = model.get_neg_is_pos(net_output)
        target = model.get_targets(sample, net_output)
        #if torch.isinf(logits).any():
        #    print("inf exist in student logits")
        #    import pdb
        #    pdb.set_trace()
            
        # teacher -- not updated
        with torch.no_grad():
            existing_masks = {"mask_indices":net_output["mask_indices"], "mask_channel_indices":net_output["mask_channel_indices"]}
            existing_neg_idxs = net_output["neg_idxs"]
            teacher_net_output = self.teacher_model(**teacher_sample["net_input"], existing_masks=existing_masks, existing_neg_idxs=existing_neg_idxs)
            teacher_logits = self.teacher_model.get_logits(teacher_net_output).float()
            teacher_mod_logits = self.teacher_model.get_mod_logits(teacher_net_output).float()
            teacher_neg_is_pos = self.teacher_model.get_neg_is_pos(teacher_net_output)
            teacher_target = self.teacher_model.get_targets(teacher_sample, teacher_net_output)

            try:
                assert (teacher_target == target).all()
                #assert (teacher_neg_is_pos == neg_is_pos).all()
            except:
                import pdb
                pdb.set_trace()
            del teacher_target
            #if torch.isinf(teacher_logits).any():
            #    print("inf exist in teacher logits")
            #    import pdb
            #    pdb.set_trace()

        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        torch.autograd.set_detect_anomaly(True)
        if self.infonce:
            # KD with KL between teacher and student distributions
            if self.distill_with_kl:
                kldiv_loss_fct = torch.nn.KLDivLoss(reduction="sum" if reduce else "none")
                # apply log_softmax on the logits before KL
                # logits[1:][neg_is_pos]

                student_dist = F.log_softmax(logits, dim=-1)
                print('studentdis', student_dist.shape)
                print('neg_is_pos', neg_is_pos.shape)
                student_dist[:,1:][neg_is_pos] = float(1e-20)
                teacher_dist = F.softmax(teacher_logits, dim=-1)
                print('teacher_dis', teacher_dist.shape)
                teacher_dist[:,1:][teacher_neg_is_pos] = float(1e-20)
                loss = kldiv_loss_fct(
                        student_dist,
                        teacher_dist,
                        ) 
                print('loss', loss)
                if torch.isinf(loss).any():
                    import pdb
                    pdb.set_trace()
                #loss = torch.mean(loss)

            # KD with CE between 
            elif self.distill_with_ce or self.distill_with_ce_mix:
                _, soft_target = torch.max(teacher_mod_logits, keepdim=False, dim=-1) # retrieve the soft-target from teacher dist
                # below, we break down cross-entropy loss to log_softmax and nll-loss
                #log_softmax = torch.nn.LogSoftmax(dim=-1)
                #student_dist = log_softmax(logits)
                #mod_student_dist = student_dist.clone() # for avoiding in-place modification during backprop (https://github.com/NVlabs/FUNIT/issues/23)
                #mod_student_dist[student_dist == -float("Inf")] = 0 # ignore neg_is_pos
                #print(mod_student_dist)
                #nll_loss = torch.nn.NLLLoss(reduction="sum" if reduce else "none")
                #loss = nll_loss(mod_student_dist, soft_target)

                teacher_loss = F.cross_entropy( # CE loss with teacher soft target 
                    logits, # do not ignore neg_is_pos since soft_target can be neg_is_pos
                    soft_target,
                    reduction="sum" if reduce else "none",
                )

                data_loss = F.cross_entropy( # original CE loss
                    mod_logits, # ignored neg_is_pos by assigning them to -inf
                    target,
                    reduction="sum" if reduce else "none",
                )
                if self.distill_with_ce:
                    loss = teacher_loss 
                elif self.distill_with_ce_mix:
                    loss = teacher_loss + data_loss
            else: 
                logging.info("No KD Loss is specified!")
                exit()
            
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target.float(),
                weights,
                reduction="sum" if reduce else "none",
            )

        sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)

            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()

                logging_output["correct"] = corr
                logging_output["count"] = count

        if log_pred:
            logging_output["logits"] = logits.cpu().numpy()
            logging_output["target"] = target.cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "correct",
            "count",
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(
                    logging_outputs
                )
                if k.startswith("loss"):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
