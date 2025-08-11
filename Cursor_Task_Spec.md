# 给 Cursor 的任务书: 
资料范围：从头到尾仔细读astra-sim和chakra repositroy:
Astra-sim 官方教程：https://astra-sim.github.io/astra-sim-docs/index.html
Chakra（固定版本）：https://github.com/astra-sim/chakra/tree/214f2c559c10f897bcc395f8e1502d80d14f1541

现在阶段只做设计（design doc），不改代码。

# 目标（MVP）
在 Astra-sim Analytical backend 中设计“拓扑重构（topology reconfiguration）”的最小可行实现，满足：
1. 在 et_def.proto 的 CommunicationType 中新增 RECONFIGURATION = 10。
2. 支持在 trace（如由 Chakra MessageToJson 生成的 dev.0.json）中定义重构事件（comm_type = 10）。
3. 重构时可切换拓扑（Ring / FullyConnected / Switch），并可同时更新带宽（bandwidth）与时延（latency）。
4. 支持重构延迟（reconfiguration delay ≥ 0，单位：cycles）；延迟内暂停通信结束后恢复。

范围限定：仅面向 Analytical network；忽略 ns-3 / Garnet。
不改接口优先：尽量重用现有拓扑加载/解析（JSON/YAML parser），避免大改类层次。



# 交付物（只交文档）
面向新手、中文说明，包含：
1. 总览：从读取 trace → 事件调度（event queue）→ 网络拓扑生效的调用链与关键模块（Astra-sim Analytical + Chakra）。
2. 改动触达面（设计层面）：需触达的文件/类/函数清单（如 extern/graph_frontend/chakra/proto/et_def.proto、System::*、TopologyManager::*、NetworkParser::*、trace 入口），并说明“为什么要改/调用”。
3. MVP 路线（最小改动）与第二阶段可选优化（如抽 ReconfigurationManager、更细粒度 per-link）。
4. 事件与格式规格、内部流程、测试与验收、边界处理（见下）。

# 重构事件设计（comm_type = 10）
同步语义
所有 rank 在同一逻辑点插入该事件（等价 barrier）。
触发后系统进入 pause，新通信不再发起；到期执行切换并应用参数，随后 resume。

延迟语义
delay_cycles：整数、cycles（与 Analytical 全局时钟一致），≥ 0。
时间到达：执行拓扑切换与参数覆盖 → 恢复通信。




# 推荐方案：Profile 驱动（trace 只指向 profile，细节放 network.yml）
network.yml 组织方式

profiles:
  - name: ring_50g_500ns
    topology: [ Ring ]
    npus_count: [ 4 ]
    bandwidth: [ 50.0 ]   # GB/s
    latency:   [ 500.0 ]  # ns

  - name: fc_200g_800ns
    topology: [ FullyConnected ]
    npus_count: [ 4 ]
    bandwidth: [ 200.0 ]
    latency:   [ 800.0 ]

  - name: switch_100g_600ns
    topology: [ Switch ]
    npus_count: [ 4 ]
    bandwidth: [ 100.0 ]
    latency:   [ 600.0 ]

active_profile: ring_50g_500ns
reconfig_default_delay_cycles: 0   # 可选


Trace JSON（沿用 Chakra 的 attr 扁平风格）
（每个 Node 是独立 JSON，对应你的现有格式）
{
  "id": "100",
  "name": "COMM_NODE",
  "type": "COMM_COLL_NODE",
  "dataDeps": ["99"],
  "attr": [
    { "name": "is_cpu_op",       "boolVal": false },
    { "name": "comm_type",       "int64Val": "10" },            // RECONFIGURATION
    { "name": "target_profile",  "stringVal": "fc_200g_800ns" },
    { "name": "delay_cycles",    "int64Val": "1000" }
  ]
}

# 应用优先级（从高到低）
1. 事件内覆盖：bandwidth_GBps / latency_ns
2. target_profile 中的 bandwidth/latency
3. active_profile 初始值（若事件未指定 target_profile）

# 内部事件流（step-by-step）
1. 读到 RECONFIGURATION → System::scheduleReconfig(node)。
2. 事件队列 push：ReconfigStart @ now、ReconfigEnd @ now + delay_cycles。
3. ReconfigStart：networkPaused = true（阻止新通信）。
4. ReconfigEnd：TopologyManager::switchToProfile(target_profile)（复用现有解析/构建），若事件内提供 bandwidth_GBps/latency_ns 则全局覆盖；networkPaused = false（唤醒等待发送）。

MVP：允许在途消息自然完成（建议 trace 在重构点前尽量无在途）。可选增强：在重构点 drain。

# 单位与校验
bandwidth_GBps：浮点，GB/s（与 network.yml 保持一致）。
latency_ns：整数，ns。
delay_cycles：整数，cycles。
校验失败（无该 profile、值非法、npus 不一致）→ 报错并终止。


# 测试与验收（不写代码，只给步骤）
1. 基线：无重构 trace，记录通信统计。
2. 重构：在计算/通信之间插入事件（如 Ring → FullyConnected，delay_cycles=1000，可附带 bandwidth_GBps 覆盖）。
3. 预期：
重构窗口内无新通信开始；
重构后新拓扑与带宽/时延生效（通过日志/统计对比基线）。



# 代码触达面（仅设计，不改）
extern/graph_frontend/chakra/proto/et_def.proto：加 RECONFIGURATION=10。
System::*：scheduleReconfig()、事件处理、networkPaused。
TopologyManager::*：switchToProfile(name)（复用现有解析与构建）。
NetworkParser::*：启动时读取 profiles[] + active_profile；提供获取/切换接口。
Trace 入口：解析 attr 中 target_profile / delay_cycles / 覆盖项。

# 边界规则
target_profile 的拓扑与当前相同：只应用带宽/时延覆盖与延迟。
delay_cycles = 0：仍触发 Start/End 两个事件，但无等待。
缺失必需字段或 profile 不存在：报错并终止（或在文档中给出你建议的默认策略、但请明确取舍）。

# 文档风格
全文中文，术语保留英文（括号标注，如 topology、event queue、bandwidth、latency、barrier）。
以新手可照着实现为目标：写清“为什么这样做”“涉及哪些函数/文件”“输入输出与单位”。
