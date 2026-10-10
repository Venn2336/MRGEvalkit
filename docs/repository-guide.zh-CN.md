# 代码导航与运行说明

核对版本：`3149cbf4ef77ddb3c6cbd3532b73e35280c031b6`。以下结论来自该版本源码的静态检查。原 ZIP 归档现已展开为 `models/` 和 `utils/` 目录，源码内容保持一致，克隆后无需解压。

## 主调用链

`eval_chunked.sh → eval.py → init_llm(args) → prepare_benchmark(...) → load_data() → eval()`。

| 文件 | 职责 |
| --- | --- |
| `eval.py` | CLI 参数、随机种子、环境变量、模型和数据集初始化 |
| `LLMs.py` | `LLMRegistry` 注册模型；模型模块按选择延迟导入 |
| `benchmarks.py` | 将数据集名称映射到适配器 |
| `models/base_llm.py` | 模型公共接口；具体实现位于模型子目录 |
| `utils/base_dataset.py` | 调用 `generate_outputs`、保存结果、分片合并与指标调用 |
| `utils/<数据集>/<数据集>.py` | 读取标注、构造影像与提示词、导出报告 |
| `utils/question_formats.py` | 报告生成提示词 |

## 模型名以注册表为准

上游 README 展示的名称不全等同于 `--model_name`。主要模型对应如下：

| README 名称 | 实际 CLI 注册名 |
| --- | --- |
| HuatuoGPT | `Huatuo` |
| HuluMed_Qwen2 | `Hulumed_qwen2` |
| HuluMed_Qwen3 | `Hulumed_qwen3` |
| InternVL | `InternVL` |
| MedGemma | `MedGemma` |
| Qwen2_5_VL | `Qwen2.5-VL` |
| Qwen3_VL | `Qwen3-VL` |

`models/Qwen3_VL/` 仅包含 HF 实现；注册器所引用的 `Qwen3_VL_vllm.py` 不在已发布源码中。选择该模型时需要使用 `--use_vllm False`，但这不代表其余依赖和运行路径已通过验证。

## 数据集与输入

| CLI 名称 | 当前读取的 JSON 路径 |
| --- | --- |
| `IU_XRAY` | `./iu_xray_test.json` |
| `MIMIC_CXR` | `./mimic_test.json` |
| `CheXpert_Plus` | `./chexpert_val.json` |
| `Amos` | `/path/to/your/model/AMOS/test_inference.json` |
| `CT-RATE` | `/path/to/your/model/CT-RATE/validation_reports_output_add_image.json` |
| `Merlin` | `/path/to/your/model/Merlin/output/test.json` |
| `RadGenome_Brain_MRI` | `/path/to/your/model/RadGenome-Brain-MRI-for-MRG/final_test.json` |
| `Ultrasound` | `/path/to/your/model/Ultrasound/ultrasound_data/final_test_combined_multilingual_test.json` |

这些加载器直接指定路径，因此仅修改 `--datasets_path` 或将其设置为 `hf` 不能自动解决数据准备问题。

以 IU_XRAY 为例，JSON 顶层是样本列表，`image` 是图像路径列表，参考报告从 `conversations[1]['value']` 读取：

```json
[
  {
    "image": ["/absolute/path/to/image.png"],
    "conversations": [
      {"from": "human", "value": "Generate a report."},
      {"from": "gpt", "value": "Reference report text."}
    ]
  }
]
```

这是根据读取代码构造的格式示例，不是实际数据。MRI 加载器则从 `video[0]` 读取切片目录；其他适配器应逐一核对。

## 输出

- 单分片：`<output_path>/<dataset>/results.json`、`metrics.json`。
- 多分片：先产生 `results_<chunk_idx>.json`，再由源码内的合并逻辑汇总。
- 数据集 `cal_metrics()` 导出 `study_id,report` 两列的预测和参考 CSV；文件名因适配器而异，例如 IU_XRAY 是 `predictions.csv` / `ground_truth.csv`，MRI 带分片后缀。
- 当前 `metrics.json` 的返回内容主要是要求使用 `cal_report_metrics.py` 的占位提示，不能当作论文 AtomiMed 得分。
- `eval.py` 中 `total_results.json` 的汇总代码已被注释。

## 运行前必须处理的问题

| 优先级 | 问题与位置 | 影响 / 下一步 |
| --- | --- | --- |
| P0 | `utils/__init__.py` 导入 MedXpertQA、MMMU 等未提供模块 | 导入 `benchmarks` 即会受阻；应收敛为已发布的 8 个数据集导出，或补齐模块 |
| P0 | 未提供 `cal_report_metrics.py`，未找到完整原子事实评分流程 | 需获取作者后续发布，或另行实现并明确与官方实现的区别 |
| P0 | 数据 JSON 和影像资源未随仓库发布，且路径写死 | 准备对应数据，并使各加载器接受实际配置路径 |
| P1 | `eval_chunked.sh` 定义 `GPT_MODEL`，传入的却是 `JUDGE_GPT_MODEL` | 若未额外定义后者，将传入空模型名；需统一变量名 |
| P1 | README 模型名称与注册名不一致 | 按上表设置 CLI 参数 |
| P1 | Qwen3-VL 的 vLLM 适配文件缺失 | 使用已提供的 HF 路径或补齐实现 |
| P1 | `utils/base_dataset.py` 依据文件数量判断分片完成，无锁或完成标记 | 存在读取尚未写完文件、重复合并及清理竞争的风险；首次验证采用单分片 |
| P2 | 未知数据集返回 `None`，`eval.py` 随后直接调用 `load_data()` | 应在初始化后提前校验；默认 `MMMU` 也不在当前分发表内 |

上述问题仅记录，没有在本次整理中改变源码。

## 环境与启动参考

依赖清单包含 `torch==2.7.0`、`vllm==0.9.0.1`、CUDA 12 的 NVIDIA 包及 `cupy-cuda12x`，应在适配的 Linux / NVIDIA GPU 环境单独配置。当前 Mac 本地目录用于阅读和整理，未验证推理。上游建议 Python 3.11+；依赖文件中的 Python / PyTorch 兼容性注释不是本次验证结论。

完成上述阻塞项并准备环境后，可从单 GPU 参数模板开始：

```bash
python -m pip install -r requirements_qwen3.txt
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --eval_datasets IU_XRAY \
  --model_name MedGemma \
  --model_path /absolute/path/to/medgemma-checkpoint \
  --output_path ./eval_results/smoke \
  --use_vllm False \
  --num_chunks 1 --chunk_idx 0 \
  --max_image_num 1 --max_new_tokens 1024 \
  --use_llm_judge False
```

这只是参数模板，未经端到端运行。原始脚本不会自动限制样本量，应先准备少量样本的输入 JSON。检查生成报告和参考报告是否逐条对齐，再扩大规模；完整论文评分仍依赖尚未提供的评分模块。
