# Loss–Miner Compatibility

下表基于 `pytorch-metric-learning` 实测输出维度（见 `losses/config.py` 中的 `MINER_OUTPUT_SHAPE`）。`expected_tuple_len` 是 loss 期望/推荐的 indices_tuple 长度；只允许白名单中的 miner，且 miner 输出的 tuple 长度必须与 loss 期望一致。

| Loss (key)           | expected_tuple_len | 允许的 Miners (输出维度)                 | 备注 |
|----------------------|--------------------|-----------------------------------------|------|
| `triplet` / `tripletmargin` | 3 | `batch_hard`(3), `triplet_margin`(3), `distance_weighted`(3) | Triplet indices `(a,p,n)` |
| `contrastive`        | 4 | `pair_margin`(4), `multi_similarity`(4), `batch_easy`(4) | Pair/quad形式 `(a1,p,a2,n)` |
| `multisimilarity` / `multi_similarity` | 4 | `multi_similarity`(4), `pair_margin`(4), `batch_easy`(4) | 推荐与自身 `multi_similarity` miner 搭配 |
| `ntxent` / `instance`| 4 | _无_（通常不用 miner） | Sup/self-supervised 对比 |
| `supcontrastive`     | 4 | _无_（通常不用 miner） | 监督对比，不建议外置 miner |

**Miner 输出维度（来自 `MINER_OUTPUT_SHAPE`）**
- `batch_hard`: 3
- `triplet_margin`: 3
- `distance_weighted`: 3
- `batch_easy`: 4
- `pair_margin`: 4
- `multi_similarity` / `multisimilarity`: 4

> 如果在 loss 中传入 `indices_tuple`，会严格检查长度；若同时使用 miner，会先 miner，再与 `indices_tuple` 做交集过滤，长度也必须一致。
