# FSDP + LoRA + SGLang：问题、上游修复、我们的修复、验证

背景

| 编号 | 类型 | 作者 | 状态 | 对应层 | 结论与依据 | 给上游的反馈要点 |
|---|---|---|---|---|---|---|
| [#7289](https://github.com/verl-project/verl/issues/7289) | Issue | jackychris | open | 4 | **问题成立**。连它标为「疑似同族、未证实」的 `wake_up()` 不对称都被实测证实，而正式修它的 #7339 完全没碰这处 | issue 不涉合并。值得回帖说明那处猜测是对的 |
| [#7290](https://github.com/verl-project/verl/issues/7290) | Issue | jackychris | open | — | **问题成立**。但它是对着 `87c68d2` 写的，键列表比现在多一个 `exclude_modules` | 倾向补全 megatron 侧的 `peft_type`，同时把必需键与枚举/字符串约定写进基类 docstring。**已按此修完（`b805cebc`）并在真机验证**；评审时提醒键列表已过期，免得按旧快照判断 |
| [#7288](https://github.com/verl-project/verl/pull/7288) | PR | jackychris | open | 1 | **值得原样采用**。方向也对：把 `all-linear` 译成 SGLang 的哨兵 `all` 交由其自行展开，而不是在 verl 侧展开成具体模块名 | 可直接合入，无异议 |
| [#7287](https://github.com/verl-project/verl/pull/7287) | PR | jackychris | open | 2、3 | **一半值得**。缺陷 1（`asdict()` 修在消费侧）理由充分；缺陷 2（请求字段）在 sglang 0.5.8 上正确，但 0.5.18 已把字段改回「每个 TP rank 一份」的列表，**修法方向与上游演进相反**，会随版本失效 | 缺陷 1、3 可直接合入；缺陷 2 建议改为对齐同文件里权重同步路径的写法（新版字段已改名为 `serialized_named_tensors: List[bytes]` 且回到 per-TP-rank 列表）。另外 `wrap_lora_params` 的类型标注要一并改 |
| [#7339](https://github.com/verl-project/verl/pull/7339) | PR（修 #7289） | shotsan | open，CLA 未签 | 4 | **不建议原样合并**。判据选错——用了第一次同步之后才置位的 `sleep_level`，正好挡不住第一次，所以只打它仍然 `KeyError: 'weights'`，且栈从它没守的 `wake_up()` 出来 | 建议改为从配置谓词推导 `sleep_level`，并一并修 `wake_up()` 的不对称（#7289 已提及）。CLA 待签 |
| 无编号 | 缺陷 | **无人报告** | — | 5、6、7 | 叠完上面四个补丁后才暴露，修复均已端到端验证 | 建议单独提 PR。问题 7 只在 megatron 路径暴露，且与问题 1 同根因、另一处现场 |

---

## 先说结论

四个 issue / PR 里能走到的点，真机上都复现了。四个补丁叠完，同一条路径上又多出三个它们没写到的缺陷。
这条路径没有端到端跑通过，所以每层都要等上一层修好才看得到下一层。

verl 现有的 5 个 LoRA 示例全部用 vllm，没有 sglang。

按执行顺序：

| 层 | 问题 | 引擎 | 上游状态 |
|---|---|---|---|
| 1 | `target_modules` 被拆成单个字符（启动参数） | 两者 | #7288 已修 |
| 2 | `asdict()` 作用在 dict 上 | 两者 | #7287 已修 |
| 3 | 请求字段名/类型与 sglang 不符 | 两者 | #7287 修法对 0.5.8 正确、对 0.5.18 失效 |
| 4 | resume 了从未 release 的 `weights` | 两者 | #7289 报告，#7339 **修不完整** |
| 5 | LoRA 权重停在 CPU，CUDA-IPC 只认设备张量 | FSDP | **无人报告** |
| 6 | 权重名带 vLLM 特有的 `.base_layer` | FSDP | **无人报告** |
| 7 | `target_modules` 被拆成单个字符（适配器配置） | megatron | **无人报告** |

前六层修完，**FSDP 路径连续跑通 4 步**。第 7 层是切到 megatron 后才暴露的：
#7290 的 peft_config 形状问题（缺 `peft_type`、rank 读错字段）修好之后，
失败点往下移了一层，落在问题 1 的同一个根因上——同一个字符串，另一处现场。

**环境**：verl main `060bebc5` + 三个 PR 分支，sglang 0.5.18.dev0，torch 2.13.0+cu130，
peft 0.19.1，A100-80G。注意 PR 作者的环境是 sglang **0.5.8**，第 3 层的分歧正源于此。

**关于单测**：#7288 和 #7287 各自带了一个 CPU 单测文件，我们开发期也加过一个契约测试，
三者都**没有进入最终提交栈**。理由是这条链上的缺陷全在 verl 与 SGLang 的线上契约里——
字段名、张量位置、参数命名、sleep/resume 记账——不起真服务就碰不到，mock 层面的测试
只是在断言 mock 的形状（#7339 就是这种情况：单测全绿，真机第一次同步即崩）。
证据因此全部由端到端运行承担，见 `validation.md`。

---

## 问题 1：`target_modules="all-linear"` 被拆成单个字符

### 背景：`target_modules` 是什么，`"all-linear"` 又写在哪

LoRA 不训练原模型的权重，而是只在**挑中的那几层**旁边挂一对小矩阵去学增量。
「挑中哪几层」就是 `target_modules`。PEFT（HuggingFace 的 LoRA 实现）允许两种写法：

- **列出模块名**，如 `["q_proj", "v_proj"]`，按名字后缀匹配；
- **写字符串 `"all-linear"`**，这是 PEFT 的简写，意思是「模型里所有线性层都挂」，
  具体是哪些由 PEFT 自己扫模型决定。

verl 的默认值就是后者，定义在配置 dataclass 里（`verl/workers/config/model.py:126`）：

```python
target_modules: Optional[Any] = "all-linear"  # allow both "all-linear" and ["q_proj","k_proj"]
```

对应到命令行是 `actor_rollout_ref.model.target_modules`，展开后的 yaml 里就是
`target_modules: all-linear`（见 `verl/trainer/config/_generated_ppo_trainer.yaml:414`）。
换句话说，**不显式指定的话，每一个 LoRA 任务都会带着这个字符串跑**。

推理侧的 SGLang 也需要知道该给哪些层准备 LoRA 显存，它的对应参数叫
`--lora-target-modules`。所以 verl 起 SGLang 服务时要把这个配置转发过去。

### 问题

verl 把 `"all-linear"` **原样**转发给了 SGLang 的 `lora_target_modules`
（`async_sglang_server.py:330`）。SGLang 拿到后做了这么一步
（`sglang/srt/server_args.py:9043`，函数 `check_lora_server_args()`）：

```python
lora_target_modules=set(self.lora_target_modules),
```

它预期这里是一个**模块名的列表**，`set()` 只是去重。但传进来的是**字符串**，
而 Python 对字符串做 `set()` 是**按字符迭代**的：

```python
set(["q_proj", "v_proj"])   # {'q_proj', 'v_proj'}      <- 预期
set("all-linear")           # {'a','l','-','i','n','e','r'}  <- 实际
```

于是 SGLang 认为要给七个名叫 `a`、`l`、`-`、`i`、`n`、`e`、`r` 的模块准备显存，
去查其中某个的隐藏层维度时炸掉：

```
NotImplementedError: get_hidden_dim not implemented for i
```

报错里那个 `i` 就是 `"all-linear"` 里的字母 i。这条信息完全指不到 `target_modules`，
所以从错误现场很难反推到配置——这也是它能存活的原因。

### 原始修复（#7288）

新增 `sglang_lora_target_modules()`，把 `all-linear` 译成 SGLang 自己的哨兵值 `"all"`，
由 SGLang 内部展开。对正则形式的 `target_modules` 直接报错而不猜。

### 我们的修复

无。**#7288 是对的，直接采用。**

它没有在 verl 侧展开成七个投影名，而是交给 SGLang，这个判断正确——SGLang 会按后端能力裁剪
（默认 `csgmv` 后端处理不了 `embed_tokens`/`lm_head`），verl 不该镜像这种后端相关逻辑。

### 验证方式

在本环境的 sglang 上直接跑机制：

```
sglang 合法模块名: ['down_proj', 'embed_tokens', 'gate_proj', 'gate_up_proj', 'k_proj', ...]
set('all-linear') = ['-', 'a', 'e', 'i', 'l', 'n', 'r']
  其中合法的: []          -> 全是单字符，无一合法
  'i' 在拆分结果里: True   -> 即报错信息里那个 i
  'all' 是哨兵不在词表: True
```

端到端侧：打上补丁后 SGLang 引擎正常启动并完成 CUDA graph capture。

---

## 问题 2：`asdict()` 作用在 dict 上

### 问题

训练走完一步，要把刚更新的 LoRA 交给 SGLang。交的不只是权重，还有一份「这个 adapter
长什么样」的配置：rank 是多少、挂在哪些层。这份配置叫 `peft_config`，放进 HTTP 请求的
`config_dict` 字段。HTTP 只能带普通 dict，不能带 Python 对象，所以对象必须先摊平。

摊平这件事，训练引擎已经做过了。基类把 `get_per_tensor_param` 的第二个返回值标成 `dict`
（`base.py:151`）：

```python
def get_per_tensor_param(self) -> tuple[..., Optional[dict]]:
```

FSDP 按这个约定做：手里的 `LoraConfig` 对象先 `.to_dict()`，交出的已经是 dict
（`fsdp/transformer_impl.py:1032`）。

SGLang 这边的 `wrap_lora_params` 拿到这份配置后，却又调了一次 `dataclasses.asdict()`：

```python
def wrap_lora_params(self, peft_config: LoraConfig, weights):
    peft_config_json = asdict(peft_config)   # 作者以为手里还是对象
    req = LoadLoRAAdapterFromTensorsReqInput(config_dict=peft_config_json, ...)
```

`asdict()` 的作用就是把 dataclass 对象的字段抠出来，做成普通 dict：

```python
from dataclasses import dataclass, asdict

@dataclass
class Demo:
    r: int
    target_modules: list

cfg = Demo(r=32, target_modules=["q_proj", "v_proj"])
asdict(cfg)
# {'r': 32, 'target_modules': ['q_proj', 'v_proj']}
```

作者调它，是因为参数标注写成了 `peft_config: LoraConfig`，整条函数按「手里还是对象」来写。
FSDP 交出来的已经是 dict。`asdict()` 只认 dataclass 实例，对 dict 直接报错：

```
TypeError: asdict() should be called on dataclass instances
```

FSDP + LoRA + SGLang 每次同步都会撞上。

整条链上各站实际拿到的类型：

| 位置 | 实际类型 | 谁转的 |
|---|---|---|
| FSDP 引擎内部 | `LoraConfig` 对象 | — |
| `get_per_tensor_param` 返回值 | `dict` | FSDP 的 `.to_dict()` |
| `wrap_lora_params` 入参 | `dict`（标注却写成 `LoraConfig`） | 原样往下传 |
| HTTP `config_dict` | 必须是 `dict` | 消费侧又 `asdict()` 一次 → 炸 |

两个函数名字像，都是「变成 dict」，输入不同：

| 调用 | 谁提供 | 输入 | 输出 |
|---|---|---|---|
| `LoraConfig.to_dict()` | PEFT | `LoraConfig` 对象 | `dict` |
| `dataclasses.asdict()` | Python 标准库 | dataclass 实例 | `dict` |

生产侧已经摊过一次，消费侧再摊就是对 dict 调一个只认对象的函数。
标注写成 `LoraConfig`，和基类的 `dict`、和 FSDP 实际交出的类型都矛盾，
静态检查看不出来，这个错就留了下来。

同文件的对照：vLLM rollout 按 dict 收，不调 `asdict`。该改的是 SGLang 这一侧。

### 原始修复（#7287 缺陷 1）

改消费侧：抽出 `normalize_peft_config_for_sglang()`，按 dict 处理；参数标注改成 `dict`。
不再调用 `asdict()`。

### 我们的修复

无。**#7287 这一条是对的，直接采用。**

基类约定的是 dict，vLLM 也按 dict 收。消费侧跟生产侧对齐即可。

### 验证方式

对照三处：基类返回类型、FSDP 的 `to_dict()`、`wrap_lora_params` 的入参。
端到端侧：这一层不再报错，流程推进到了下一层。

---

## 问题 3：请求字段名与类型和 sglang 不符

### 问题

训练走完一步，要把刚更新的 LoRA 小矩阵交给 SGLang，推理才用得上新 adapter。
交的方式是构造一个请求对象，再走 HTTP 发给 SGLang 进程。

张量不能直接塞进 HTTP，要先序列化成一串 bytes。请求里就有一个字段专门装这串 bytes。
这个字段**叫什么、类型是什么、要几份**，是 SGLang 定的，verl 必须对上。

SGLang 推理侧常开 tensor parallel（TP）：模型切到多张卡上，每张卡一个 rank。
有的版本要求「每个 rank 一份」序列化结果，列表长度等于 `tp_size`；
有的版本只要一份，由 SGLang 自己广播。内容可以完全一样——循环变量 `i` 用不上，
只是把同一份 blob 重复 N 次，好让每个 rank 各拿一个槽位。

verl main 当时这样写：

```python
req = LoadLoRAAdapterFromTensorsReqInput(
    lora_name=...,
    config_dict=...,
    serialized_tensors=[serialize(weights) for _ in range(tp_size)],  # 字段名 + 列表
)
```

两处都可能对不上：字段名是不是 `serialized_tensors`，值该是列表还是单个字符串。

### 原始修复（#7287 缺陷 2）

作者的环境是 sglang **0.5.8**。那个版本里这个字段的类型是 `str`：只要一份序列化结果，
不要列表。于是他删掉循环，改成发单个字符串：

```python
req = LoadLoRAAdapterFromTensorsReqInput(
    serialized_tensors=serialize(weights),  # 还是旧字段名，值改成一份 str
)
```

他同时指出那个循环是死代码（循环体不用 `i`，只是把同一字符串重复 N 次）——
**在 0.5.8 上这点成立**：服务端只要一份，自己会分发给各 rank，列表长度没有意义。

字段名他没改，0.5.8 用的也还是 `serialized_tensors`。

### 我们的修复

**照抄同文件里已经能跑的全量权重同步。**（commit `956267a7`）

本环境是 sglang **0.5.18**。同一个请求结构体相对 0.5.8 改了三处：

```python
class LoadLoRAAdapterFromTensorsReqInput(BaseReq, kw_only=True):
    lora_name: str
    config_dict: Dict[str, Any]
    # One serialized copy of the adapter tensors per TP rank
    serialized_named_tensors: Annotated[List[bytes], Base64Bytes()]
```

| 变化 | 0.5.8（#7287 的环境） | 0.5.18（本环境） |
|---|---|---|
| 字段名 | `serialized_tensors` | `serialized_named_tensors` |
| 类型 | `str`（一份） | `List[bytes]`（每个 TP rank 一份） |
| 实现 | pydantic | msgspec |

注释写着「每个 TP rank 一份」。0.5.8 删掉的那个列表，0.5.18 又要回来了。
所以在 0.5.18 上，verl main 和 #7287 **字段名都错**；#7287 还把类型改成了单份 `str`，
和 0.5.18 的列表也不符。

同文件里，更新基座权重的那条路径一直能跑，写法是：

```python
req = UpdateWeightsFromTensorReqInput(
    serialized_named_tensors=[MultiprocessingSerializer.serialize(named_tensors) for _ in range(tp_size)],
    ...
)
```

0.5.18 上这两个请求的字段名和类型一样。LoRA 路径按同一形态构造：

```python
serialized_named_tensors = [MultiprocessingSerializer.serialize(weights) for _ in range(tp_size)]
req = LoadLoRAAdapterFromTensorsReqInput(
    lora_name=...,
    config_dict=...,
    serialized_named_tensors=serialized_named_tensors,
)
```

两个调用点从此一致。列表里 N 份内容相同，是因为这条路径上 adapter 没有按 TP 切分，
每个 rank 反序列化同一份即可；**长度**仍要等于 `tp_size`，否则对不上 rank 下标。

#7287 对 0.5.8 是对的。但它会随版本失效：0.5.18 改回了 per-TP-rank 列表，
而且还换了字段名。修法方向和上游后来的演进相反，评审时值得提出来。

### 验证方式

在真实 sglang 0.5.18 上构造请求，三种写法对照：

| 传参 | 结果 |
|---|---|
| `serialized_tensors=...`（verl main 与 #7287 都用这个名字） | `TypeError: Unexpected keyword argument` |
| `serialized_named_tensors=[b"..."]`（我们的写法） | 成功，tp=1/2/4 均通过 |

verl main 在 sglang ≥ 0.5.18 上这条路径本身就是断的，与这三个 PR 无关——
字段名已经对不上，还没走到类型那一步。

---

## 问题 4：resume 了从未 release 的 `weights`

### 问题

`rollout.mode=async` 只决定生成怎么发（走 SGLang HTTP 服务），不决定卡怎么分。
本实验是 hybrid：训练引擎和 SGLang 融在同一组 GPU 上，所以 rollout 跑完要 `sleep()`
腾显存给训练，下一步同步权重之前再 `resume()` 把腾出去的东西要回来。
SGLang 用一个 set 记下卸了哪些东西，叫 `offload_tags`。
`resume` 从里面 `remove(tag)`。`remove()` 是严格操作：tag 不在集合里就抛 `KeyError`。

能卸的就两样：`kv_cache`（推理缓存）和 `weights`（基座权重）。

adapter 模式（`lora.merge=False`）只同步 LoRA 小矩阵，基座权重一直留在 GPU 上。
所以 `sleep()` 只该卸 `kv_cache`，不该卸 `weights`。merge 模式 / 非 LoRA 才两个都卸。

崩的现场是：`sleep()` 只卸了 `kv_cache`，`offload_tags` 里没有 `weights`，
却有人来 `resume(tags=["weights"])`：

```
KeyError: 'weights'
  ... scheduler_update_weights_mixin.py, in resume_memory_occupation
      self.offload_tags.remove(tag)
```

谁来 resume，有两处，都可能要错 tag。

### 原始修复（#7339）

只挡了其中一处：`engine_workers.py` 在 resume 前看 `sleep_level`。

```python
if getattr(self.rollout, "sleep_level", 2) != 1:
    await self.rollout.resume(tags=["weights"])
```

约定是：`sleep_level=1` 表示 adapter 模式，只卸过 `kv_cache`，不要去要 `weights`；
`sleep_level=2` 表示卸过两样，可以要。

#7339 完全没碰另一处 `wake_up()`。而且它用的 `sleep_level`，第一次同步时还是错的，见 4b。

### 我们的修复

**#7339 不完整，实测打上后仍然崩在同一个 `KeyError`。** 两处都要补。

提交栈在这一层有三条：与 #7339 等价的那个守卫（`9ee0073c`），外加下面两条它没覆盖的。
那个守卫本身是必需的——`ServerAdapter.resume(tags)` 把 tags 直接透传给 SGLang，
而 adapter 模式的 `release()` 只卸 `kv_cache`，所以去掉它，`engine_workers` 那条路径照样崩。

**4a. `wake_up()` 不对称**（commit `4c0a1d01`）

`sleep()` 已经按配置分支了：

```python
# async_sglang_server.sleep()
tags = ["kv_cache"] if self.lora_as_adapter else ["kv_cache", "weights"]
```

`wake_up()` 的 COLOCATED 分支当时不分支，写死恢复两样：

```python
# 修之前
obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
```

释放侧分了、恢复侧没分。adapter 模式下 `sleep()` 没卸 `weights`，`wake_up()` 却去要。
改成与 `sleep()` 对称：

```python
tags = ["kv_cache"] if self.lora_as_adapter else ["kv_cache", "weights"]
```

#7289 提到过这处不对称，标为「疑似同族、未证实」。#7339 完全没碰。实测它是真的。
colocated 主循环当前不走 `wake_up()`（见 `validation.md` 的 `f2_wake_up`），
这条是 standalone / 以后接上 `wake_up()` 时会撞的同一类错。

**4b. `sleep_level` 的时序**（commit `e4ecaf5a`）

这是 colocated 主循环第一次同步就崩的原因，也是 #7339 挡不住的原因。
卸什么、要不要 resume `weights`，两处各看各的，而且生效时间不同：

| 判据 | 谁在用 | 看的是什么 | 何时为真 |
|---|---|---|---|
| `lora_as_adapter` | `sleep()` / `wake_up()` | 配置：rank>0 且 `merge=False` | **第 0 步就为真** |
| `sleep_level` | `engine_workers` 的 resume | 一个整数，初值 2 | **第一次同步之后**才被写成 1 |

`sleep_level = 1` 写在第一次同步的后半段（`engine_workers.py:780`）。
#7339 用它做门闩，第一次同步时它还是 2，门是开的。

于是第一次同步的顺序是：

1. `sleep()` 看配置，只卸 `kv_cache`，`offload_tags = {kv_cache}`
2. `engine_workers` 看 `sleep_level`，还是 2，去 `resume(tags=["weights"])`
3. SGLang `remove("weights")` → `KeyError`

修法：构造时就按同一个配置函数赋值，不要等第一次同步再写。

```python
self.sleep_level = 1 if lora_served_as_adapter(self.model_config) else 2
```

三处从此看同一件事，第一次迭代也对。

### 验证方式

端到端跑 Qwen3-0.6B adapter 模式：

- 只打 #7339 → 仍然 `KeyError: 'weights'`。#7339 守住的是 `engine_workers` 那条，
  且第一次同步时 `sleep_level` 还是 2，门闩不起作用
- 补上 4a + 4b → `KeyError` 消失，SGLang 完成 CUDA graph capture，推进到下一层

---

## 问题 5：LoRA 权重停在 CPU，CUDA-IPC 只认设备张量

**上游无人报告。** 修好前四层之后才会到达。

### 问题

```
IndexError: tuple index out of range
  ... sglang/srt/utils/patch_torch.py, in _modify_tuple
```

`collect_lora_params()` 以 `.detach().cpu()` 结尾——**故意**把权重放 CPU 以降低峰值显存。
而 SGLang 走 CUDA IPC 传权重，torch 的 reducer 只为设备张量生成设备槽位。
sglang 的 `patch_torch` 把索引 6 硬编码为设备位，注释还写着「签名多年没变，用常量看起来安全」。

### 我们的修复（commit `51fb107d`）

在上线前逐个把 CPU 张量搬回设备，两条路径（基座同步 + adapter 同步）都加。
逐个搬保留了原本的省显存意图——整个 state dict 不会同时驻留在显存上。

CPU 暂存本身是合理的选择，只在**上线那一刻**是错的，所以在那里转换。

### 验证方式

直接量归约元组长度：

| 张量位置 | 归约元组长度 | 序列化 |
|---|---|---|
| CUDA | 15（索引 6 有效） | 成功 |
| CPU | 3（索引 6 越界） | `IndexError: tuple index out of range` |

补丁后 `IndexError` 消失，流程推进到第 6 层。

---

## 问题 6：权重名带 vLLM 特有的 `.base_layer`

**上游无人报告。** 修好前五层之后才会到达。

### 问题

基座权重第一次同步时，SGLang 的 loader 按名字查参数。查不到就炸：

```
KeyError: 'model.layers.0.self_attn.qkv_proj.base_layer.weight'
```

这个名字是训练引擎改过的。FSDP 在交出基座权重前，对每个 key 跑一遍
`replace_lora_wrapper()`（`fsdp_utils.py:780`）。函数名像是在剥包装，实际是**往名字里插一段**：

```
q_proj.weight  →  q_proj.base_layer.weight
```

docstring 写着「for proper weight loading in vLLM」。vLLM 给挂了 LoRA 的层加一层包装，
冻结的原权重放在子模块 `base_layer` 里，loader 要带着这段才能对上。

SGLang 没有这层包装。它把 q/k/v 融合成 `qkv_proj`，权重直接挂在这个模块上，
登记的名字是 `qkv_proj.weight`。多出来的 `.base_layer` 一路带进 loader，对不上任何一项。

| 侧 | 它认的基座权重名 |
|---|---|
| 训练引擎交出（给 vLLM 改过的） | `...qkv_proj.base_layer.weight` |
| vLLM loader | `...qkv_proj.base_layer.weight` |
| SGLang loader | `...qkv_proj.weight` |

改名发生在训练引擎里。引擎不知道下游是 vLLM 还是 SGLang，两条 rollout 收到同一套带
`.base_layer` 的名字。

### 我们的修复（commit `b1c6db71`）

在 SGLang 上线前把这段去掉：

```python
# sglang_rollout.py
name.replace(".base_layer.", ".")   # qkv_proj.base_layer.weight → qkv_proj.weight
```

改名是为某个 rollout 做的，却由不区分后端的引擎施加。在有分歧的那一侧撤销，
vLLM 路径完全不动。让引擎按 rollout 后端分支是更干净的做法，属于架构改动，不放进这次 bugfix。

### 验证方式

端到端跑通基座同步：这个 `KeyError` 消失，流程推进到下一层。

---

## 问题 7：`target_modules="all-linear"` 被拆成单个字符（第二处现场）

**上游无人报告。只在 megatron 路径上暴露。**

### 问题

和问题 1 同一个字符串、同一种拆法，现场不同。问题 1 坏在**启动 SGLang 时**的
`--lora-target-modules`；这里坏在**每次加载 adapter 时**，请求体里那份 `peft_config`。

启动参数那边 #7288 已经译成哨兵 `"all"`，引擎能起来。第一次
`load_lora_adapter_from_tensors` 时，SGLang 拿 adapter 自带的 `target_modules`
去和内存池做兼容检查，400 拒掉：

```
LoRA adapter verl_actor_lora_name with rank 32 is incompatible with the current
LoRA memory pool configuration. Please ensure that the LoRA adapter's rank is within
the configured `--max-lora-rank` and that the target modules are included in
`--lora-target-modules`.
```

这段正文默认是看不到的。`http_server_engine` 里 `raise_for_status()` 抛出的异常只带状态行，
例如 `400 Client Error: Bad Request for url: .../load_lora_adapter_from_tensors`。
后面的 `except HTTPError as e` 也只打 `e`，响应体就被丢掉了。
所以在 raise 之前先读 body 打一行：

```python
if response.status_code >= 400:
    _log_error_body(endpoint, response.status_code, response.text)
response.raise_for_status()
```

aiohttp 那条更要紧：`await response.text()` 会把流读完，`async with` 一退出 body 就没了，
必须在 `raise_for_status()` 之前读。没有这个判断，现场只剩一个 400，对不上问题 7。

这条信息并列了 rank 和模块名，没说是哪个触发的。`can_support()` 实际有三道门
（`mem_pool.py:247`）：rank、added tokens、模块名。rank 和 added tokens 实测都过
（`32 > 32` 为假、`0 > 0` 为假）。坏的是模块名：

```
adapter.target_modules=['a','l','l','-','l','i','n','e','a','r']
pool.target_modules=['down_proj','embed_tokens','gate_up_proj','lm_head','o_proj','qkv_proj']
```

SGLang 的 `get_normalized_target_modules()` 认得字符串 `"all-linear"`，会译成哨兵 `{"all"}`。
一旦已经是列表，就走另一条分支：拿列表里每一项去和池子求子集。十个单字母没有一个在池子里。

拆它的是 `normalize_peft_config_for_sglang()` 里一行无条件的 `list()`，#7287 加的。
这行本身有用：FSDP 走 `LoraConfig.to_dict()`，`target_modules` 是 `set`，JSON 序列化不了，
必须先变成 list。megatron 走 `build_peft_config_for_vllm()`，这个字段是字符串 `"all-linear"`
（注释还写着「vLLM 并不真的用 target_modules，放个占位」）。

`list()` 对两种输入的结果：

| 生产侧 | 交出的 `target_modules` | `list(...)` 之后 |
|---|---|---|
| FSDP `to_dict()` | `{"q_proj", "v_proj", ...}`（set） | `["q_proj", "v_proj", ...]` |
| megatron `build_peft_config_for_vllm()` | `"all-linear"`（str） | `['a','l','l','-','l','i','n','e','a','r']` |

同一行代码，FSDP 过、megatron 拆。FSDP 全绿时这个缺陷藏得住。

### 我们的修复

字符串原样留下，只有 set / list 才转成 list：

```python
target_modules = normalized["target_modules"]
normalized["target_modules"] = target_modules if isinstance(target_modules, str) else list(target_modules)
```

`"all-linear"` 穿过之后，SGLang 自己译成 `{"all"}`，和问题 1 启动参数那侧一致。

### 验证方式

端到端：megatron + LoRA + SGLang 跑满 12 步，适配器加载零拒绝，其中 4 步有真实梯度，
rollout 与训练的 log-prob 没有分叉。数据见 `validation.md`。

开发期另有 CPU 单测核过 `"all-linear"` 原样穿过，以及两个断言 megatron 配置**没有**
`peft_type` 的旧用例——补上这个键正是 #7290 的内容，所以它们改成了断言新的约定。
这些单测最终没有进入提交栈，理由见「先说结论」末尾。

---

## 复现方式

只跑端到端。下面是真机跑通时的环境；PR 作者用的是 sglang **0.5.8**，第 3 层的字段分歧正源于此。

| 组件 | 版本 / 配置 |
|---|---|
| verl | main `060bebc5` + #7288 / #7287 / #7339 + 本文修复（当前栈顶 `9ee0073c`） |
| sglang | 0.5.18.dev0 |
| torch | 2.13.0+cu130 |
| peft | 0.19.1 |
| megatron-core | 镜像自带；TransformerEngine、flash-attn 按 torch 2.13 重建后可 import |
| GPU | 单节点 4×A100-80G，hybrid 同卡，cgroup 内存上限 128 GB |
| 布局 | `rollout.mode=async`，`rollout.tensor_model_parallel_size=2`，`gpu_memory_utilization=0.5` |
| 模型 / 数据 | Qwen3-0.6B / GSM8K |

两边共用的故障链开关：

| 配置 | 作用 |
|---|---|
| `rollout.name=sglang` | 触发问题 1（引擎启动） |
| `lora.merge=False`（默认） | adapter 模式，触发问题 2/3/5/6 |
| `free_cache_engine=True` | 启用 sleep/resume 状态机，触发问题 4 |
| `total_training_steps>=3` | 问题 4 的一部分要第二次同步才暴露 |

### FSDP

问题 1–6 在这条路径上验的。默认 `ppo_trainer.yaml`（`model_engine=dp`），LoRA 走扁平字段
`lora_rank` / `lora_alpha` / `target_modules`。矩阵里的 50 步长稳、TP 1/2/4、merge / 非 LoRA
非回归也是同一套命令。

```bash
#!/usr/bin/env bash
# FSDP + LoRA + SGLang end-to-end. Walks the failure chain (issues 1-6).

set -xeuo pipefail

REPO=${REPO:?path to the verl checkout carrying the patch stack}
VENV=${VENV:?path to the python env with verl + sglang installed}
ASSETS=${ASSETS:?directory holding the model and the dataset}

MODEL_PATH=${MODEL_PATH:-$ASSETS/model/Qwen/Qwen3-0.6B}
DATA=${DATA:-$ASSETS/testset/gsm8k}
STEPS=${STEPS:-4}
NGPUS=${NGPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
EXP=${EXP:-lora_sglang_adapter}
RUNLOG=${RUNLOG:-$PWD/run_logs/${EXP}.log}
mkdir -p "$(dirname "$RUNLOG")"
[[ -s $RUNLOG ]] && mv "$RUNLOG" "${RUNLOG%.log}.$(date -r "$RUNLOG" +%m%d_%H%M).log"

export PYTHONPATH=$REPO
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TMPDIR=${TMPDIR:-/tmp}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torch_compile}
export GLOO_SOCKET_IFNAME=${IFACE:-eth0} NCCL_SOCKET_IFNAME=${IFACE:-eth0}
unset RAY_ADDRESS

cd "$REPO"

"$VENV/bin/python3" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger='["console"]' \
    trainer.project_name=lora_fix_acceptance \
    trainer.experiment_name="$EXP" \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps="$STEPS" \
    ${EXTRA_ARGS:-} \
    "$@" 2>&1 | tee "$RUNLOG"
```

### Megatron

问题 7 在这条路径上才暴露。已在上表环境跑通 12 步。必须 `--config-name=ppo_megatron_trainer`，
否则默认 FSDP 树会拒掉 `actor.megatron`。LoRA 读的是 `model.lora.rank`，不是 FSDP 那组扁平的
`model.lora_rank`。

```bash
#!/usr/bin/env bash
# megatron + LoRA + SGLang end-to-end -- the pairing #7290 is about.
#
# This is the case the issue's reporter could not test ("I have no megatron
# setup"), and it only became runnable after rebuilding TransformerEngine and
# flash-attn against torch 2.13: the image ships both compiled against the
# system torch 2.11, and megatron.core imports TE at module scope.
#
# Note the config key: megatron reads `model.lora.rank` (a dict block) while FSDP
# reads the flat `model.lora_rank`. Those two blocks are never synced, which is
# half of why the peft_config shapes drifted in the first place.
#
# Without the #7290 fix this dies in SGLang's adapter loader, which rejects a
# peft_config with no `peft_type` -- so this run doubles as that fix's negative
# control if you revert it.

set -xeuo pipefail

# Point these at your own checkout, env and asset root.
REPO=${REPO:?path to the verl checkout carrying the patch stack}
VENV=${VENV:?path to the python env with verl + sglang installed}
ASSETS=${ASSETS:?directory holding the model and the dataset}

MODEL_PATH=${MODEL_PATH:-$ASSETS/model/Qwen/Qwen3-0.6B}
DATA=${DATA:-$ASSETS/testset/gsm8k}
STEPS=${STEPS:-4}
NGPUS=${NGPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ACTOR_TP=${ACTOR_TP:-1}
EXP=${EXP:-lora_sglang_megatron}
RUNLOG=${RUNLOG:-$PWD/run_logs/${EXP}.log}
mkdir -p "$(dirname "$RUNLOG")"
[[ -s $RUNLOG ]] && mv "$RUNLOG" "${RUNLOG%.log}.$(date -r "$RUNLOG" +%m%d_%H%M).log"

export PYTHONPATH=$REPO
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TMPDIR=${TMPDIR:-/tmp}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torch_compile}
export GLOO_SOCKET_IFNAME=${IFACE:-eth0} NCCL_SOCKET_IFNAME=${IFACE:-eth0}
unset RAY_ADDRESS

cd "$REPO"

# The megatron.* config block only exists in this config tree; the default
# (FSDP) one rejects `actor.megatron` as an unknown key.
"$VENV/bin/python3" -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora.rank=32 \
    actor_rollout_ref.model.lora.alpha=32 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$ACTOR_TP" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.logger='["console"]' \
    trainer.project_name=lora_fix_acceptance \
    trainer.experiment_name="$EXP" \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps="$STEPS" \
    ${EXTRA_ARGS:-} \
    "$@" 2>&1 | tee "$RUNLOG"
```

---
