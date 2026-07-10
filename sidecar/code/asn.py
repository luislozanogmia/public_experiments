#!/usr/bin/env python3
# Provenance: Shannon Son / Hamming lab. ASN (Associative/Attractor Synaptic Network) class -- a Hopfield-style network built to sit outside a frozen LLM and steer it. Dependency of exp_d_asn_pilot.py. Source: experiments/hamming/asn.py
"""
ASN - Artificial Synaptic Network.

A Hopfield attractor network that learns by OBSERVING an LLM,
never by co-training with it.

Phase A (learning): LLM processes text. ASN watches hidden states.
                    ASN forms attractors via Hebbian. LLM untouched.
Phase B (steering): ASN recalls attractor for new input.
                    Injects as context. LLM generates. LLM untouched.

The ASN is the mind. The LLM is the mecha.
The pilot learns the machine by watching it work.
Then the pilot flies it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ASN(nn.Module):
    """
    Artificial Synaptic Network.

    A Hopfield attractor network operating in the LLM's hidden state space.
    Learns from observed hidden states, not from text directly.

    Architecture:
    - n_neurons: total neurons in the ASN
    - d_state: dimension of the LLM's hidden states (768 for GPT-2)
    - W_intra: (n_blocks, block_size, block_size) - attractor weights
    - proj_in: maps LLM hidden state -> ASN neuron space
    - proj_out: maps ASN neuron space -> LLM hidden state space
    """

    def __init__(self, d_state=768, n_neurons=4096, n_blocks=16,
                 k_winners=200, settle_steps=5, beta=1.0):
        super().__init__()
        self.d_state = d_state
        self.n_neurons = n_neurons
        self.n_blocks = n_blocks
        self.block_size = n_neurons // n_blocks
        self.k_winners = k_winners
        self.settle_steps = settle_steps
        self.beta = beta

        # Projection: LLM space <-> ASN neuron space
        # These are learned during the observation phase
        self.proj_in = nn.Linear(d_state, n_neurons, bias=False)
        self.proj_out = nn.Linear(n_neurons, d_state, bias=False)

        # Hopfield interaction weights (the brain)
        self.W_intra = nn.Parameter(
            torch.randn(n_blocks, self.block_size, self.block_size) * 0.01
        )
        self.gate_bias = nn.Parameter(torch.zeros(n_neurons))

        # Init projections small
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ASN: {n_params/1e6:.1f}M params")
        print(f"  {n_neurons} neurons, {n_blocks} blocks x {self.block_size}")
        print(f"  d_state={d_state}, k={k_winners}, settle={settle_steps}")

    def _k_wta(self, x):
        _, indices = torch.topk(x.abs(), self.k_winners, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, indices, 1.0)
        return x * mask

    def _hopfield_step(self, state):
        """One settle step: block-diagonal interaction + k-WTA."""
        # state: (B, N) where N = n_neurons
        B = state.shape[0]
        state_blocks = state.view(B, self.n_blocks, self.block_size)
        W_sym = (self.W_intra + self.W_intra.transpose(-1, -2)) / 2
        field = torch.einsum('bnk,nkj->bnj', state_blocks, W_sym)
        field = field.reshape(B, self.n_neurons)
        new_state = torch.tanh(self.beta * (field + self.gate_bias))
        return self._k_wta(new_state)

    def settle(self, llm_hidden):
        """
        Take an LLM hidden state, project into ASN space,
        settle to attractor, project back to LLM space.

        Input: (B, d_state) - an LLM hidden state
        Output: (B, d_state) - the ASN's "thought" in LLM space
        """
        # Project into neuron space
        h = self.proj_in(llm_hidden)  # (B, n_neurons)
        state = self._k_wta(h + self.gate_bias)

        # Settle to attractor
        for _ in range(self.settle_steps):
            state = self._hopfield_step(state)

        # Project back to LLM space
        return self.proj_out(state)  # (B, d_state)

    def observe_and_learn(self, hidden_states, lr=1e-3):
        """
        Hebbian learning from observed hidden states.
        No backprop. No loss function. Pure attractor formation.

        Input: hidden_states (B, d_state) - extracted from LLM
        """
        with torch.no_grad():
            # Project LLM states into ASN neuron space
            h = self.proj_in(hidden_states)  # (B, n_neurons)
            pre = self._k_wta(h + self.gate_bias)

            # Settle
            state = pre.clone()
            for _ in range(self.settle_steps):
                state = self._hopfield_step(state)
            post = state

            # Hebbian update: strengthen connections for post-settle co-activation
            pre_blocks = pre.view(-1, self.n_blocks, self.block_size)
            post_blocks = post.view(-1, self.n_blocks, self.block_size)

            # Mean over batch
            pre_mean = pre_blocks.mean(dim=0)   # (n_blocks, block_size)
            post_mean = post_blocks.mean(dim=0)  # (n_blocks, block_size)

            total_delta = 0.0
            for n in range(self.n_blocks):
                delta = (torch.outer(post_mean[n], post_mean[n])
                        - torch.outer(pre_mean[n], pre_mean[n]))
                delta = (delta + delta.T) / 2
                delta.fill_diagonal_(0)
                self.W_intra.data[n] += lr * delta
                total_delta += delta.abs().mean().item()

            # Decay to prevent runaway
            self.W_intra.data *= 0.9999

            return total_delta / self.n_blocks

    def learn_projections(self, hidden_states, lr=1e-4):
        """
        Learn proj_in and proj_out so that:
        project_in -> settle -> project_out ~= original hidden state

        This is reconstruction learning: the ASN learns to encode
        LLM hidden states as attractors and decode them back.
        Still NO backprop through the LLM - just through ASN itself.
        """
        self.train()
        h_in = self.proj_in(hidden_states)
        state = self._k_wta(h_in + self.gate_bias)
        for _ in range(self.settle_steps):
            state = self._hopfield_step(state)
        reconstructed = self.proj_out(state)

        # Reconstruction loss: can the ASN round-trip the hidden state?
        loss = F.mse_loss(reconstructed, hidden_states.detach())
        return loss, reconstructed

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'd_state': self.d_state,
                'n_neurons': self.n_neurons,
                'n_blocks': self.n_blocks,
                'k_winners': self.k_winners,
                'settle_steps': self.settle_steps,
            }
        }, path)
        print(f"  ASN saved: {path}")

    @classmethod
    def load(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        asn = cls(**cfg).to(device)
        asn.load_state_dict(ckpt['state_dict'])
        print(f"  ASN loaded: {path}")
        return asn
