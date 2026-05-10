# Task-Aware Sensitivity Heatmap

This table shows the performance drop (delta) when specific modules are ablated. **More negative = Higher Sensitivity**.

| Module | Overall Delta | BN Delta | EN Delta | HI Delta |
|---:|---:|---:|---:|---:|
| `model.layers.0.mlp` | **🔴 -45.27** | **🔴 -29.93** | **🔴 -77.32** | **🔴 -28.57** |
| `model.layers.0.self_attn` | **🔴 -25.82** | **🔴 -28.91** | **🔴 -23.71** | **🔴 -24.83** |
| `model.layers.13.self_attn.k_proj` | 🟠 -3.88 | 🟠 -6.12 | 🟠 -6.53 | 🟢 +1.02 |
| `model.layers.4.self_attn` | 🟠 -3.30 | 🟠 -6.80 | 🟠 -3.09 | 0.00 |
| `model.layers.4.mlp` | 🟠 -2.97 | 🟠 -2.04 | 🟠 -6.53 | -0.34 |
| `model.layers.18.mlp.down_proj` | 🟡 -1.36 | 🟡 -1.70 | -0.34 | 🟠 -2.04 |
| `model.layers.27.mlp.up_proj` | 🟡 -1.25 | 🟠 -3.74 | -0.34 | 🟢 +0.34 |
| `model.layers.18.self_attn.o_proj` | 🟡 -0.91 | 🟡 -1.36 | 0.00 | 🟡 -1.36 |
| `model.layers.22.mlp.gate_proj` | 🟡 -0.68 | 🟡 -0.68 | 🟡 -1.37 | 0.00 |
| `model.layers.13.mlp.act_fn` | -0.46 | 🟠 -3.06 | 🟡 -0.69 | 🟢 +2.38 |
| `model.layers.27.self_attn.v_proj` | 🟢 +0.00 | 🟢 +0.34 | 🟢 +1.03 | 🟡 -1.36 |
| `model.layers.22.self_attn.q_proj` | 🟢 +0.23 | 🟢 +0.68 | 🟢 +0.69 | 🟡 -0.68 |
