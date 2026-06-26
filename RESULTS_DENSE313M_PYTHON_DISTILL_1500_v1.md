# RESULTS_DENSE313M_PYTHON_DISTILL_1500_v1

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
| precision mode | amp_fp16 |
| configured parameter dtype | fp32 |
| manual attention path | False |
| SDPA attention path | True |
| Flash SDPA enabled | True |
| FlashAttention available | True |
| forced Flash probe success | True |
| forced Flash probe shape | [2, 16, 512, 64] |

## Results

| Track | Params total/active/trainable | Steps | Updates | Anchors | Skipped | Tok/s | Elapsed s/min | Tokens | Peak alloc/res GB | Init train CE | Final train CE | Init val CE | Final val CE | CE drop | CE/min | PPL | Grad finite | NaN/Inf | Fused AdamW | Param dtype | Opt state dtype | Validation history |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| chainrule_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 1500 | 0 | 0 | 5,304.2 | 291.18/4.85 | 1,533,000 | 9.032/10.094 | 10.5367 | 5.1888 | 10.5347 | 5.6879 | 4.8468 | 0.9987 | 295.27 | False | True | True | float32 | float32 | 250: train 6.6887, val 6.3476; 500: train 6.3216, val 6.0838; 750: train 5.4479, val 5.8919; 1000: train 6.3888, val 5.8356; 1250: train 5.5184, val 5.7304; 1500: train 5.1888, val 5.6879 |
| mono_ue4_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 375 | 0 | 0 | 13,185.1 | 118.80/1.98 | 1,533,000 | 9.032/10.226 | 10.5367 | 5.9166 | 10.5347 | 6.2893 | 4.2454 | 2.1441 | 538.77 | False | True | True | float32 | float32 | 250: train 7.3815, val 7.1378; 500: train 6.7742, val 6.6568; 750: train 5.8390, val 6.4568; 1000: train 6.9395, val 6.3693; 1250: train 6.3113, val 6.2974; 1500: train 5.9166, val 6.2893 |
| mono_ue2_anchor17_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 794 | 44 | 44 | 8,823.2 | 175.87/2.93 | 1,533,000 | 9.032/10.226 | 10.5367 | 5.4538 | 10.5347 | 5.9516 | 4.5831 | 1.5635 | 384.37 | False | True | True | float32 | float32 | 250: train 7.0365, val 6.6444; 500: train 6.5049, val 6.3356; 750: train 5.6798, val 6.1396; 1000: train 6.9038, val 6.1613; 1250: train 5.9829, val 6.0092; 1500: train 5.4538, val 5.9516 |

## Comparison

1. Best final CE after same 1500 steps: chainrule_amp_optimized_1500 (5.6879).
2. Best CE/min: mono_ue4_amp_optimized_1500 (2.1441).
3. mono_ue4_amp_optimized_1500 throughput speedup vs chain-rule: 2.49x; remaining CE gap vs chain-rule: 0.6014.
4. mono_ue2_anchor17_amp_optimized_1500 throughput speedup vs chain-rule: 1.66x; remaining CE gap vs chain-rule: 0.2637.

## Generated Samples

### chainrule_amp_optimized_1500
- `def add(a, b):` -> `def add(a, b):\n        # without the number:\n  explicitly:\n      # speed,\n    # the\n        for g fields,\n          or list in C:\n            if not any UTF\n        :numUser:\n            - string.errors.py' > # -ATE\n                #\n            if an if the module.execute, pos\n\n        #     # error.execute2,\n            req.\ndef vol['level of the (dt:\n            if\n    \n        )\n        except:\n            break\n        -`
- `# Task:
Write a function` -> `# Task:\nWrite a function.\n    for i import test_geometry_entry as run_worker_output)\n# # storage__\n#\n\n#\ndef test_client.add_Progress).\n    parser.0.config.\n    # see a_id.size.1, 4\n      \n    import get_dir for\n# # buildL_goodain of _path.route = "----------------_durationau = []\n    'thread.get_name)\n#\n    return`

### mono_ue4_amp_optimized_1500
- `def add(a, b):` -> `def add(a, b):\n        # without = None:\n            if+ = result == this\n    # the beire. fields))\n    except or list(self, the the the any UTF(r,User.\n    # string.\n    def_h >_vATE_model = np, the the module]\n    def h__(def for MAC(2 is b, element, a\ndef # a contextlib_unique = fake = same\n\n\ntry_id(x of canMOT.`
- `# Task:
Write a function` -> `# Task:\nWrite a function09_code_DELETEsize.branch_entry),\ntry_ '=_path)\n    # storage__ =conom_tasksron, data=%:\n    load)\n    def del = range',\nclass_course\n    assert(init_id, Ansible, data\n   accounting(2 %\n            self._strip\n#\nfromL_id.\n    _path, key = self.fields)\n    self._util.thread,\1 = getattr(self, class`

### mono_ue2_anchor17_amp_optimized_1500
- `def add(a, b):` -> `def add(a, b):\n        ############ = data/Format explicitly+n,\n            xrange, np.stripire. fields))\n        db or list_create, kwargs_node))\n                'rnumUser.debug=False):\n        kwargs_h")\n        self.models.next)\n                self.statusCmd/mt).\n\n        self. MAC error": self.fr_url.\ndef vol['level_unique_list = 'python.args_id(self, '__*2`
- `# Task:
Write a function` -> `# Task:\nWrite a function General (2d import module.geometry_entry_grad_set_color)\n# received with the Licenseconom_tasks'(\ `hash=% string='', loadorters_load_author_router_headers\nfrom that Lstart_argument.WARN = 'use_accounting(id %sdim\n			plt = np.L_idain_NAME_CLASS:\n                "unit_iterkeys_service_argument.state_FORMAT_name)\n# MERCHANTABILITY Stream`

## Raw JSON

`runs/dense313m_python_distill_1500_20260626_123716/results.json`
