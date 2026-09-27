# Aligned endpoint foresight and motion–hand candidate evaluation

Branch: feat/action-grounded-alignment. Base: feat/music-jepa-action-prior at 4fcad2085b48de80b2109ab3866575b4b9b68c2e. This is an opt-in implementation for the next paper revision. The spatial worktree and original action-prior branch are unchanged.

## Configuration

Merge scripts/config/overrides/aligned_foresight.yaml into the actual action-prior training configuration, then supply framework.delta_jepa.prediction_horizon=h and framework.delta_jepa.executed_prefix=K. The ??? entries require a choice based on the client's command rate and buffering. Require 1 <= K <= h <= H and h >= 2. Start with h=K; longer lookahead is supported. Save the resolved config in the training run.

The initial aligned implementation supports the lerobot_datasets loader, one ego camera (num_video_views: 1, duplicate_single_view: false), future-only chunks, first-command inverse labels, and current proprioception. Other loaders/camera configurations are rejected rather than silently using shifted-video targets. The example group split [64,14] is SONIC-specific. A different controller requires its real coordinate grouping; do not apply this split to the 36D interface.

The legacy_shift default preserves the old visual construction. Model weights are not automatically converted into a trained endpoint evaluator. A warm start may reuse compatible tensors, but the new temporal meaning and candidate conditioning require training; record initialization and missing/unexpected keys.

## Training contract

- Fetch frames indexed 0...h and commands 0...H-1 from the same demonstration start using recorded frame timestamps.
- Exclude samples that would pad either the future target or proposal action labels beyond episode end.
- Use the same PIL resize path for dataset endpoints, online observations, and subgoals (resolution_size, then optional image_size). Encode repeated current and repeated endpoint images separately with a frozen encoder; use last-block spatial features for both.
- One predictor block has access to every one of the h ordered action tokens. Sinusoidal action-token position encoding retains sequence order; the original per-step MLP/interpolation alone would lose order under a single symmetric attention block.
- Train dense endpoint alignment, pooled feature alignment, shared first-command reconstruction, policy flow matching, and the temporal prior.
- Both visual and prior evaluation use the first h commands, with the same prefix in prior training. The proposal still has H commands.

## Joint and factorized controls

prior_type: joint retains a shared GRU. prior_type: factorized creates one model per command group, with the same state but only that group's action history. The subgroup width is selected to match the joint parameter budget within 1% for the reference SONIC dimensions. Both use the same command_group_weights: [64,14] is the original all-coordinate mean; [1,1] is group-balanced.

Use separate held-out validation to tune beta for joint, factorized, first-difference, and second-difference energies, with the same search budget. The implementation does not claim those weights were tuned.

## Inference and records

The request includes use_verifier=True, executed_prefix=K, and optionally return_candidate_trace=True. The response carries execution_contract; the client must actually execute only K commands. The server's check validates the declared K, not the robot's physical execution.

Trace includes every candidate's visual progress, prior energy, group residuals, total score, and selected index, as well as actual observation/goal and configuration identity. Candidate generation remains sequential. Goal and current pooling match the endpoint target.

For replay scoring, supply candidate_actions in the evaluator's normalized coordinates and candidate_normalization_id matching its configured normalization_id. This skips sampling. Use starVLA/tools/candidate_evaluation.py to preserve canonical controller commands through explicit affine conversions; also preserve state normalization, coordinate order, and controller/decoder hashes. tools/recording/save_decision.py checks and stores the canonical command bank and affine map. Normalization IDs are provenance checks, not proof of correct conversion.

A fixed-candidate bank needs independently measured Q for every replayed candidate. Restore simulator, controller recurrent state, and RNG state; use an independent stage event or the same continuation policy. selection_metrics reports selected Q and candidate-set regret and rejects missing outcomes. No simulator restoration API or real replay is fabricated here.

## Validation and status

Run python -m pytest tests/test_aligned_foresight.py -q in a PyTorch environment. The local CPU run passes 16 tests. Tests cover unpadded dataset indices, endpoint isolation, the production endpoint-encoding/predictor paths with small test doubles, ordered full-prefix action access, actual block-mask construction, factorized-history isolation and pairing identity, group-weight equivalence, parameter matching, affine command identity, measured-outcome regret, preprocessing consistency, unevaluated-tail invariance, and trace round trips/integrity. They do not instantiate the full VLM/video checkpoint or measure robot performance.

No GPU training, full checkpoint inference, WebSocket integration, or closed-loop robot/SIMPLE rollout has been run in this workspace. Existing paper numbers, including 9.5 Hz, remain R0 measurements. R1 measurement slots remain blank.


## Preparing a resolved training configuration

Use OmegaConf.merge(base, override) in the existing training environment and set prediction_horizon and executed_prefix to the measured command-count choices. Resolve the result with OmegaConf.to_container(config, resolve=True, throw_on_missing=True) before saving: leaving either ??? unset must fail. The SONIC example base is scripts/config/vlajepa_sonic_latent_vjepa21.yaml; its action_horizon is 40. This does not determine the executed prefix K.

Launch the resolved YAML with the existing accelerate/deepspeed command in scripts/vlajepa_sonic_latent_vjepa21.sh, replacing only --config_yaml and using a new output/run directory. For the factorized comparison, start another run with prior_type=factorized and the same data, optimization budget, h/K, initialization protocol, and group weights. Do not resume optimizer state as if an R0 checkpoint had the same endpoint semantics. All data/checkpoint paths must be resolved on the actual training server.

Before the first training run, inspect a decoded dataset sample and confirm the recorded command at index t acts over the interval ending at t+1 under the collection/control protocol. Index pairing is implemented; it does not estimate camera-control timestamp offsets. For efficiency, inventory all resident modules: the inverse decoder is not called by selection but remains instantiated unless explicitly unloaded.
