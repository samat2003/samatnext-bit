# RESULTS_DENSE313M_PYTHON_HF_MIX_QUALITY_1500_v1

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

## KL Distillation

KL distillation skipped: no usable teacher logits/cache exists for the exact python_hf_mix_2p5b dataset/tokenizer/vocab. The only candidate inspected was /home/samat_zharassov/samatnext-qwen/results/samatnext_qwen_gdn8_direct/stage1_teacher_cache_20260623-161900.pt; it contains one batch of input_ids/labels, teacher_hidden_states shaped (1, 512, 1536), and teacher_loss, but no reusable vocab-32768 teacher logits.

## Results

| Track | Params total/active/trainable | Steps | Attempts | Applied updates | Warmup CR | Mono updates | Anchors | Anchor collisions | KL alpha/temp | Skipped opt | Tok/s | Elapsed s/min | Tokens | Peak alloc/res GB | Init train CE | Final train CE | Init val CE | Final val CE | CE drop | CE/min | PPL | Grad finite | Nonfinite grad events | Nonfinite grad sample | NaN/Inf | Fused AdamW | Param dtype | Opt state dtype | Scaler final | Scaler overflow/skips | Scaler checkpoints | Validation history |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|---|---:|---:|---|---|
| chainrule_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 1500 | 1500 | 0 | 0 | 0 | 0 | n/a | 0 | 5,074.0 | 304.28/5.07 | 1,533,000 | 9.032/10.031 | 10.5367 | 4.8408 | 10.5347 | 5.4794 | 5.0553 | 0.9968 | 239.70 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 250:65536.0->65536.0; 500:65536.0->65536.0; 750:65536.0->65536.0; 1000:65536.0->65536.0; 1250:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 6.5838, val 6.2100; 500: train 6.2289, val 5.9563; 750: train 5.3611, val 5.7722; 1000: train 6.1942, val 5.7218; 1250: train 5.2611, val 5.5727; 1500: train 4.8408, val 5.4794 |
| mono_ue1_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 1500 | 1500 | 0 | 1500 | 0 | 0 | n/a | 0 | 5,343.1 | 288.89/4.81 | 1,533,000 | 9.032/10.165 | 10.5367 | 4.8063 | 10.5347 | 5.4658 | 5.0689 | 1.0528 | 236.45 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 250:65536.0->65536.0; 500:65536.0->65536.0; 750:65536.0->65536.0; 1000:65536.0->65536.0; 1250:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 6.5838, val 6.2100; 500: train 6.2276, val 5.9557; 750: train 5.3547, val 5.7722; 1000: train 6.2038, val 5.7220; 1250: train 5.2616, val 5.5748; 1500: train 4.8063, val 5.4658 |
| mono_ue2_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 750 | 750 | 0 | 750 | 0 | 0 | n/a | 0 | 9,068.6 | 171.04/2.85 | 1,533,000 | 9.032/10.165 | 10.5367 | 5.2682 | 10.5347 | 5.8165 | 4.7181 | 1.6551 | 335.81 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 250:65536.0->65536.0; 500:65536.0->65536.0; 750:65536.0->65536.0; 1000:65536.0->65536.0; 1250:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 7.0186, val 6.5617; 500: train 6.4372, val 6.2311; 750: train 5.5922, val 6.0562; 1000: train 6.6129, val 5.9788; 1250: train 5.7812, val 5.8596; 1500: train 5.2682, val 5.8165 |
| warmup100_chainrule_then_mono_ue2_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 800 | 800 | 100 | 700 | 0 | 0 | n/a | 0 | 9,184.5 | 168.72/2.81 | 1,533,000 | 9.032/10.165 | 10.5367 | 5.2493 | 10.5347 | 5.8153 | 4.7193 | 1.6783 | 335.41 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 250:65536.0->65536.0; 500:65536.0->65536.0; 750:65536.0->65536.0; 1000:65536.0->65536.0; 1250:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 6.8186, val 6.3555; 500: train 6.3809, val 6.1622; 750: train 5.5789, val 5.9972; 1000: train 6.5960, val 5.9634; 1250: train 5.7735, val 5.8543; 1500: train 5.2493, val 5.8153 |
| warmup250_chainrule_then_mono_ue2_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 875 | 875 | 250 | 625 | 0 | 0 | n/a | 0 | 8,619.4 | 179.65/2.99 | 1,533,000 | 9.032/10.165 | 10.5367 | 5.2086 | 10.5347 | 5.7704 | 4.7642 | 1.5912 | 320.68 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 250:65536.0->65536.0; 500:65536.0->65536.0; 750:65536.0->65536.0; 1000:65536.0->65536.0; 1250:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 6.5838, val 6.2100; 500: train 6.3199, val 6.0874; 750: train 5.5438, val 5.9472; 1000: train 6.5517, val 5.9251; 1250: train 5.6879, val 5.8175; 1500: train 5.2086, val 5.7704 |
| warmup500_chainrule_then_mono_ue2_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 1000 | 1000 | 500 | 500 | 0 | 0 | n/a | 0 | 7,685.8 | 201.26/3.35 | 1,533,000 | 9.032/10.165 | 10.5367 | 5.1685 | 10.5347 | 5.7216 | 4.8131 | 1.4349 | 305.38 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 250:65536.0->65536.0; 500:65536.0->65536.0; 750:65536.0->65536.0; 1000:65536.0->65536.0; 1250:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 6.5838, val 6.2100; 500: train 6.2289, val 5.9561; 750: train 5.4502, val 5.8600; 1000: train 6.4855, val 5.8559; 1250: train 5.6369, val 5.7653; 1500: train 5.1685, val 5.7216 |
| mono_ue2_anchor17_amp_optimized_1500 | 369,927,168/369,927,168/369,927,168 | 1500 | 794 | 794 | 0 | 750 | 44 | 44 | n/a | 0 | 9,163.2 | 169.12/2.82 | 1,533,000 | 9.032/10.165 | 10.5367 | 5.2261 | 10.5347 | 5.7928 | 4.7419 | 1.6823 | 327.92 | True | 0 | n/a | False | True | float32 | float32 | 65536.0 | 0 | -3:65536.0->65536.0; -2:65536.0->65536.0; -1:65536.0->65536.0; 250:65536.0->65536.0; 500:65536.0->65536.0; 750:65536.0->65536.0; 1000:65536.0->65536.0; 1250:65536.0->65536.0; 1500:65536.0->65536.0 | 250: train 6.9792, val 6.5400; 500: train 6.4172, val 6.2274; 750: train 5.5632, val 6.0310; 1000: train 6.6515, val 6.0096; 1250: train 5.7454, val 5.8652; 1500: train 5.2261, val 5.7928 |

## Comparison

1. Best final CE after same 1500 steps: mono_ue1_amp_optimized_1500 (5.4658).
2. Best CE/min: mono_ue2_anchor17_amp_optimized_1500 (1.6823).
3. Main question: quality-mode mono-forward reduces the chain-rule final CE gap only if its final CE gap is smaller than the prior mono gap, and beats chain-rule only if final CE is lower under the same 1500-step conditions.
4. mono_ue1_amp_optimized_1500 throughput speedup vs chain-rule: 1.05x; remaining CE gap vs chain-rule: -0.0136.
5. mono_ue2_amp_optimized_1500 throughput speedup vs chain-rule: 1.79x; remaining CE gap vs chain-rule: 0.3372.
6. warmup100_chainrule_then_mono_ue2_1500 throughput speedup vs chain-rule: 1.81x; remaining CE gap vs chain-rule: 0.3360.
7. warmup250_chainrule_then_mono_ue2_1500 throughput speedup vs chain-rule: 1.70x; remaining CE gap vs chain-rule: 0.2911.
8. warmup500_chainrule_then_mono_ue2_1500 throughput speedup vs chain-rule: 1.51x; remaining CE gap vs chain-rule: 0.2422.
9. mono_ue2_anchor17_amp_optimized_1500 throughput speedup vs chain-rule: 1.81x; remaining CE gap vs chain-rule: 0.3134.

## Generated Samples

### chainrule_amp_optimized_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function General Public License\n#    (geometry_entry)\n# You '=_version)# # and the License\n#\n#    data groups here='', load the Free Software Foundation version without the License\n#    # If the GNU General Public License for the default: you have\n   ipv, but or **\n#    # The c in the Free Software Foundation (at the GNU General Public Licenseau to the GNU redistribute it will be the GNU General Public License as the`
- `def add(a, b):` -> `def add(a, b):\n        - (old_version[0+2,0,2,1,0*2,2,5,2,2b,5,0,0,User\2,2.7,2h,1,ATE3,0)\n---\nix7,0,2,02,02,04,\ndef2E6AC(r,2selection2292\nAx05ER*2`

### mono_ue1_amp_optimized_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function General Public License is free; without even the GNU General Public License at\n#    it to the storage the License\n#\n#    If the implied\n# A even the License version without the License\n#    If the GNU General Public License and the GNU General Public License\n       but it will be useful for\n#    to the GNU General Public License for the GNU General Public License and or theau to the GNU General Public License # If strings:\n# See the`
- `def add(a, b):` -> `def add(a, b):\n        ############ = "%sFormat = [2, Tri1.0.strip(np.namespace(value) == 0.5].y_sequence(priority.numUser.0,2.7,1h >1.ATE_model_label(1,i] +0).1,1.2.02,0,0,0,2,0.r,2,2292ff_A)  3*2`

### mono_ue2_amp_optimized_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function Generaltext):\n    import existing.geometry_entry_grad_set_Status)\n    """\n    if d =conom_tasks_match_id = np = set_Progress_author_router_headers_out = sum(return_id_size_id_accounting(id %s = i for model = np.L_id = '/NAME_id:\n                "unit_iterkeys_service_util.threadclient.urls_PYTHON_bugs = config`
- `def add(a, b):` -> `def add(a, b):\n        ############ = str(Format = [2,\n            else:\n                return\n            6. fields))\n        except or list(\n            RH(a_sequence(priority.numUser.debug=False):\n       mod_h")\n        - context_model_name))\n        elif module.args, h = 1,\n        """\n        """\n        self.get(self.assertEqual['level_unique_list =selection_display\n        x_response_volume_MOT.`

### warmup100_chainrule_then_mono_ue2_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function General _code_x_value_id:\n        if m_ '=_STORE_data_equal(lenconom_tasks_match_id:\n            load_src_URL = range_all()\n        if not None:\n            elif None:\n                for data_accounting(id %s" % **t = np.L_idain_NAME_id:\n                "unit_iterkeys_class book(\n                #client.urls_PYTHON_GROUP = config`
- `def add(a, b):` -> `def add(a, b):\n        - data = str(Formatsize+2,\n            xrange.nameACT_config. fields))\n        m_full_create_some_node_name_r_User.debug_query.errors.bind_id_vATE_model_label_set_iCmd_entry).1])\n        raise error_module2_info_mailbox_name = offset_vars_id = int(2 = flags_id(x_value*2`

### warmup250_chainrule_then_mono_ue2_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function General _code_x_value_id_time_hash_ '=_STORE_data_equal(len_line_name, data_start_target_src_URL=user_all()\nfrom_FILE_doesnt_id_size_id_accounting(id %s"\n		 temporarily\n#    if isinstance_idain_NAME_description:\n                "unit_iterkeys_service_pb-threadclient.urls_PYTHON_GROUP_key`
- `def add(a, b):` -> `def add(a, b):\n        # data = "%s[1+ = result == 0\n    # return data in g=namespace.value=None=None, kwargs_node=None,r=User=None=False.errors.bind=np_vATE_model_name='set_i]\n    def pos=None,\n        error=None=fr_url.name=False['level_unique_list = 'scale=None_id', 'volume_MOT_`

### warmup500_chainrule_then_mono_ue2_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function General _code_x_value_id:\n        if m_ '=_path_data:\n        elif fieldsconom_tasks_match_id:\n            load_src_URL = range_all()\n        if not None:\n            elif None:\n                for data_accounting(id %s = i in sorted_id)\n        c_arguments_NAME_id:\n                "unit_iterkeys_class_pb.thread_FORMAT_op_x_in_`
- `def add(a, b):` -> `def add(a, b):\n        # program = "%sFormat = [2,\n            else:\n                return data in g[0]\n        # list(self, kwargs_node))\n        self.numUser.debug_query.assertEqual(1) > 1:\n            # In_label_set_i]\n        if self._params_pypi(unittest_msg, uid_mailbox\n        # vol['level_unique_list = self.display("py_response)\n        self._set`

### mono_ue2_anchor17_amp_optimized_1500
- `# Task:
Write a function` -> `# Task:\nWrite a function General Public License import import existing.geometry_entry_grad_set_color)\n# useful and the License\n#\n# `hashHandler:\nimportorters_Progress:\n        range(this:\n            login = sum(return_id, Version\n            raise [],accounting(id %s"\n		,\n#    if isinstance(self):\n        _path.error(request(self)au_files(self):\n                # If strings.fields):\n        return`
- `def add(a, b):` -> `def add(a, b):\n        # data = np.Format = [2,\n            else:\n                return\n            6. fields.name, or list(\n            RH(y_sequence(priority.numUser.debug=False):\n       mod_h")\n        - context = np = np.set_i]\n                if h = 1, MAC(unittest=2, data=mailbox\n        return vol['level_unique_list = np.display\n        x_response)\n        break\n        self`

## Raw JSON

`runs/dense313m_python_hf_mix_quality_1500_20260626_134951/results.json`
