# RESULTS_DENSE313M_PYTHON_HF_MIX_1500_v1

Dense313M-class audit on `python_hf_mix_2p5b` tokenized Python corpus.

This is a real Python pretrain loss-mechanics audit, not teacher-logit distillation and not a coding benchmark or pass@1 evaluation.

Lower final validation CE is better. Higher CE/min means faster validation CE improvement per elapsed minute.

## Dataset

| Field | Value |
|---|---|
| dataset label | python_hf_mix_2p5b |
| symlink path | data/generated/python_hf_mix_2p5b |
| source path | /home/samat_zharassov/samatnext-qwen/data_prepared/python_hf_mix_512_2p5b |
| tokenizer path | data_prepared/python_hf_mix_512_2p5b/tokenizer.json |
| tokenizer type | bytelevel_bpe |
| vocab size | 32768 |
| train tokens | 2450000000 |
| val tokens | 50000000 |
| examples/documents | 1384374 |
| split | train.bin/val.bin |
| token dtype | uint16 |

## Optimization/Fairness

| Field | Value |
|---|---|
| active layers | 24 |
| hidden | 1024 |
| heads | 16 |
| batch/seq | 2/512 |
| learning rate | 0.0001 |
| grad clip | 1.0 |
| precision mode | amp_fp16 |
| configured parameter dtype | fp32 |
| manual attention path | False |
| SDPA attention path | True |
| Flash SDPA enabled | True |
| FlashAttention available | True |
| forced Flash probe success | True |
| forced Flash probe shape | [2, 16, 512, 64] |

## Results

| Track | Params total/active/trainable | Steps | Attempts | Applied updates | Skipped opt | Anchors | Anchor collisions | Tok/s | Elapsed s/min | Tokens | Peak alloc/res GB | Init train CE | Final train CE | Init val CE | Final val CE | CE drop | CE/min | PPL | Grad finite | Nonfinite grad events | Nonfinite grad sample | NaN/Inf | Fused AdamW | Param dtype | Opt state dtype | Scaler final | Scaler overflow/skips | Scaler checkpoints | Validation history |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|---|---:|---:|---|---|
| chainrule_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 1500 | 1500 | 0 | 0 | 0 | 5,506.6 | 280.20/4.67 | 1,533,000 | 9.032/10.031 | 10.5367 | 4.8440 | 10.5347 | 5.4793 | 5.0554 | 1.0825 | 239.67 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 250:65536.0->65536.0; 500:65536.0->65536.0; 750:65536.0->65536.0; 1000:65536.0->65536.0; 1250:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 6.5839, val 6.2101; 500: train 6.2280, val 5.9556; 750: train 5.3552, val 5.7734; 1000: train 6.2127, val 5.7207; 1250: train 5.2571, val 5.5778; 1500: train 4.8440, val 5.4793 |
| mono_ue4_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 375 | 375 | 0 | 0 | 0 | 15,816.1 | 98.75/1.65 | 1,533,000 | 9.032/10.165 | 10.5367 | 5.7236 | 10.5347 | 6.0868 | 4.4479 | 2.7024 | 440.01 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 500:65536.0->65536.0; 1000:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 7.2211, val 6.9574; 500: train 6.7165, val 6.5499; 750: train 5.7791, val 6.3498; 1000: train 6.8268, val 6.2099; 1250: train 6.1314, val 6.1559; 1500: train 5.7236, val 6.0868 |
| mono_ue2_anchor17_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 794 | 794 | 0 | 44 | 44 | 9,261.5 | 167.32/2.79 | 1,533,000 | 9.032/10.165 | 10.5367 | 5.2275 | 10.5347 | 5.7916 | 4.7430 | 1.7008 | 327.54 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 250:65536.0->65536.0; 500:65536.0->65536.0; 750:65536.0->65536.0; 1000:65536.0->65536.0; 1250:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 6.9793, val 6.5400; 500: train 6.4172, val 6.2275; 750: train 5.5632, val 6.0310; 1000: train 6.6515, val 6.0096; 1250: train 5.7458, val 5.8653; 1500: train 5.2275, val 5.7916 |

## Comparison

1. Best final CE after same 1500 steps: chainrule_amp_optimized_1500 (5.4793).
2. Best CE/min: mono_ue4_amp_optimized_1500 (2.7024).
3. mono_ue4_amp_optimized_1500 throughput speedup vs chain-rule: 2.87x; remaining CE gap vs chain-rule: 0.6075.
4. mono_ue2_anchor17_amp_optimized_1500 throughput speedup vs chain-rule: 1.68x; remaining CE gap vs chain-rule: 0.3123.

## Generated Samples

### chainrule_amp_optimized_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function General Public License is free;.  # pylint to be string with '=_version_data.\n__ = 'O_name, Inc_name='', loadorters_to_author_router_headers_out that it_argument_type_size_use_accounting(\n    %s but or **_args.api_dt_arguments of _path.\n    if not to_duration_class book(xx_client.\n    # If free field_key`
- `def add(a, b):` -> `def add(a, b):\n        ############ = "%sFormat = [n, v1.startswith(self.data[0])\n        # list(self.assertEqual(node))\n        self.numUser.debug("daily.assertEqual(1) > 1)\n        # Add_name__set_i]\n        # pos = 1\n        # error.execute(fr_url.\n        # # In contextlib_unique = 1\n        self.display("py_range)\n        #MOT.`

### mono_ue4_amp_optimized_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function General_code_DELETE_value_id_time_hash_ '=_name_data_xb(lenconom_tasks_match_id,='',_ Comments_assertEqual(self, value_headers_course_FILE_doesnt_id, np.1,accounting(id,\n            self._strip_create_api_ Safari,ain_type_description_args, 3_iterkeys_class_util.thread_FORMAT_ looks_turn_in_`
- `def add(a, b):` -> `def add(a, b):\n        # data = np"\n  """\n    data, 0,\n    # return so 6rom=namespace(value or list(self, kwargs_node))\n    retr,User, new=None):\n    def __h, i:\n    "model_name, 'limit(x, _(').1,\n                        MAC(by_msg, 0, a type=0, contextlib_unique_list = np,\n    )\n    if x)\n    '__*2`

### mono_ue2_anchor17_amp_optimized_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function General Public License import import existing.geometry_entry_grad_set_color)\n# useful and the License\n#\n# `hashHandler:\n loadorters_Progress:\n        range(this:\n            login = sum(s)\n       WARN = None, [],accounting(id)\n            if i for\n#    if isinstance(self):\n        _path.error(request(self)au = []\n        """\n        assert_urls(self, field = None`
- `def add(a, b):` -> `def add(a, b):\n        # data = np.Format = [2,\n            else:\n                return\n            6. fields.name, or list(\n            RH(y_sequence(priority.numUser.debug=False):\n       mod_h")\n        - context = np_name))\n        elif self.args, self.params_list(unittest=2, data=mailbox\n        title.train_vars(r=None,\n        try:\n                if self.volume_MOT.`

## Raw JSON

`runs/dense313m_python_hf_mix_1500_20260626_130118/results.json`
