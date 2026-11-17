# AMM-01: verified causal-probe strategy memory

Preregistered deterministic CPU pilot in [PROTOCOL.md](PROTOCOL.md).
It tests family-keyed strategy memory with synthetic rank-at-most-two recurrent operators. Agents here are frozen deterministic policies with one equal-dose causal probe per scored case; no LLM, trained BPD factorization, or live Buzz relay is run. The experiment code uses the Python standard library; importing the host BPD package requires CPU PyTorch.

Use Python >=3.11, standard library only. Run the following from the BPD repository root:

```bash
python -m unittest mechinterp.bpd.experiments.amm01_agent_memory.test_experiment -v
python -m mechinterp.bpd.experiments.amm01_agent_memory.experiment --phase development --out /tmp/amm01_development.json
# Save the development receipt's evalmath_preflight_input object as /tmp/amm01_pilot.json.
python ../EvalMath/skills/evalmath-mechinterp-power/scripts/preflight.py --pilot /tmp/amm01_pilot.json --strict --out /tmp/amm01_preflight.json
python -m mechinterp.bpd.experiments.amm01_agent_memory.experiment --phase confirm_fresh --preflight /tmp/amm01_preflight.json --out /tmp/amm01_confirm_fresh.json
```

The EvalMath script is taken from pinned EvalMath source commit `2fdd5fe3f499070b63c474a2455f0e2c00fc55a8`. A strict preflight failure blocks confirmation before any fresh-world generator access. `--check FILE` recomputes a locked receipt and refuses discrepancies. Training only writes a per-family strategy table, with a SHA-256 over its canonical JSON. World IDs and random streams are phase separated.

**Meaning of a success:** a memory policy transports a pre-selected causal-probe horizon between independent worlds in a deliberately constructed family. A new nonlinear/TRM/BPD agent experiment must be designed separately with source-disjoint training, independently defined behavior, and causal shams. The family routing signal is engineered in this pilot, so success cannot establish mechanism discovery or broader agent competence.

### Buzz bridge

The exporter creates only an unsigned NIP-29 kind-9 channel-message draft. Supply an actual channel UUID and public BPD PR URL:

```bash
python -m mechinterp.bpd.experiments.amm01_agent_memory.bridge \
  --receipt /tmp/amm01_confirmation.json \
  --channel 00000000-0000-4000-8000-000000000001 \
  --github-url https://github.com/MoralityLabAI/BPD/pull/REPLACE
```

An authorized Buzz client must sign and publish the message. The bridge exports a digest and summary, never full model labels, keys, or researcher conversations. No relay connection is opened by this code.
