#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ppo.py
@Time    :   2023/07/14 11:23:57
@Author  :   Zhou Zihao
@Version :   1.0
@Desc    :   None
'''

from .base import Base
import numpy as np
import torch
import torch.nn.functional as F

class PPO(Base):

    def __init__(self,
                 model,
                 obs_space,
                 action_space,
                 device,
                 save_path,
                 recurrent=False,
                 lr=0.001,
                 max_grad_norm=0.5,
                 adam_eps=1e-8,
                 clip_eps=0.2,
                 entropy_coef=0.05,
                 kickstarting_coef_initial=1.0,
                 kickstarting_coef_decent=0.02,
                 kickstarting_coef_minimum=0.05,
                 iter_with_ks=3000,
                 value_loss_coef=.5,
                 batch_size=128,
                 num_worker=4,
                 epoch=3,
                ):
        # model
        super().__init__(model, obs_space, action_space, device, save_path, recurrent)

        # optimizer
        self.lr                = lr
        self.grad_clip         = max_grad_norm
        self.eps               = adam_eps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, eps=self.eps)

        # loss
        self.clip              = clip_eps
        self.entropy_coef      = entropy_coef
        self.iter_with_ks      = iter_with_ks
        self.ks_coef           = kickstarting_coef_initial
        self.ks_coef_minimum   = kickstarting_coef_minimum
        self.ks_coef_descent   = kickstarting_coef_decent
        self.value_loss_coef   = value_loss_coef

        # other settings
        self.batch_size        = batch_size
        self.epochs            = epoch
        self.num_worker        = num_worker
        self.iter              = 0

    def __call__(self, obs, mask, states):
        return self.model(obs, mask, states)

    def update_kickstarting_coef(self, recent_success_rate: float = 0.0):
        """
        Decay kickstarting coefficient each iteration.

        If PPO is demonstrating competence (recent_success_rate > 0.2),
        decay 3x faster to hand control to PPO sooner.
        This implements the adaptive curriculum:
          early   → strong teacher guidance
          middle  → PPO internalising, teacher fading
          late    → PPO dominant, teacher for failures only
        """
        self.iter += 1
        if self.ks_coef <= self.ks_coef_minimum:
            self.ks_coef = self.ks_coef_minimum
        else:
            # Competence-triggered fast decay
            if recent_success_rate >= 0.3:
                decay = self.ks_coef_descent * 3.0
            elif recent_success_rate >= 0.1:
                decay = self.ks_coef_descent * 2.0
            else:
                decay = self.ks_coef_descent
            self.ks_coef = max(self.ks_coef_minimum, self.ks_coef - decay)

    def update_policy(self, buffer, recent_success_rate: float = 0.0):
        losses = []

        for _ in range(self.epochs):
            for batch in buffer.sample(self.batch_size, self.recurrent):
                obs_batch, action_batch, return_batch, advantage_batch, values_batch, mask, log_prob_batch, teacher_prob_batch = batch

                # get policy
                states = self.model.init_states(self.device, obs_batch.size()[1]) if self.recurrent else None
                pdf, value, _ = self.model(obs_batch, mask, states)

                # update
                entropy_loss = (pdf.entropy() * mask).mean()

                # Correct cross-entropy kickstarting formula
                kickstarting_loss = -(teacher_prob_batch * F.log_softmax(pdf.logits, dim=-1)).sum(dim=-1).mean()

                ratio = torch.exp(pdf.log_prob(action_batch) - log_prob_batch)
                surr1 = ratio * advantage_batch * mask
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = values_batch + torch.clamp(value - values_batch, -self.clip, self.clip)
                surr1 = ((value - return_batch)*mask).pow(2)
                surr2 = ((value_clipped - return_batch)*mask).pow(2)
                value_loss = torch.max(surr1, surr2).mean()

                # Dynamic entropy protection:
                # If entropy falls below 0.5 nats → double the bonus
                # If entropy falls below 0.2 nats → triple the bonus
                # This prevents irreversible policy collapse from a single lucky episode
                current_entropy = entropy_loss.detach().item()
                if current_entropy < 0.2:
                    eff_entropy_coef = self.entropy_coef * 3.0
                elif current_entropy < 0.5:
                    eff_entropy_coef = self.entropy_coef * 2.0
                else:
                    eff_entropy_coef = self.entropy_coef

                loss = policy_loss + self.value_loss_coef * value_loss - eff_entropy_coef * entropy_loss
                if self.iter < self.iter_with_ks:
                    loss += self.ks_coef * kickstarting_loss

                # Update actor-critic
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                # Update logger
                losses.append([loss.item(), entropy_loss.item(), kickstarting_loss.item(), policy_loss.item(), value_loss.item()])

        if self.iter < self.iter_with_ks:
            self.update_kickstarting_coef(recent_success_rate)

        mean_losses = np.mean(losses, axis=0)
        return mean_losses
