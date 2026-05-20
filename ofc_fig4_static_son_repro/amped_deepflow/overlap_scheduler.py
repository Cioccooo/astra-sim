import csv
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


BACKWARD_PRE_TP_ALPHA = 0.6
BACKWARD_POST_TP_PRE_DP_ALPHA = 0.5
BWD_COMP_PRE_TP = "BWD_COMP_PRE_TP"
BWD_COMP_POST_TP_PRE_DP = "BWD_COMP_POST_TP_PRE_DP"
BWD_COMP_POST_TP_POST_DP = "BWD_COMP_POST_TP_POST_DP"


@dataclass
class ComputeWindow:
    stage: int
    microbatch: int
    layer: int
    sublayer: str
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class DPCommWindow:
    bucket_id: int
    stage: int
    bytes: float
    ready_time: float
    raw_comm_duration: float
    raw_reconf_duration: float
    start: Optional[float] = None
    end: Optional[float] = None
    comm_start: Optional[float] = None
    comm_end: Optional[float] = None
    reconf_start: Optional[float] = None
    reconf_end: Optional[float] = None
    hidden_comm_duration: float = 0.0
    exposed_comm_duration: float = 0.0
    hidden_reconf_duration: float = 0.0
    exposed_reconf_duration: float = 0.0
    unit_kind: str = "layerwise"
    allow_pre_ready_reconfiguration: bool = True
    first_layer: Optional[int] = None
    last_layer: Optional[int] = None
    layer_count: int = 0
    topology_signature: str = "dp_none"
    configured_reconf_duration: float = 0.0
    available_local_comm_slack: float = 0.0
    request_time: Optional[float] = None
    fabric_wait_duration: float = 0.0


@dataclass
class TPCommWindow:
    layer: int
    locality: str
    ready_time: float
    raw_comm_duration: float
    request_time: Optional[float] = None
    start: Optional[float] = None
    end: Optional[float] = None
    hidden_comm_duration: float = 0.0
    exposed_comm_duration: float = 0.0
    fabric_wait_duration: float = 0.0


@dataclass
class WaitWindow:
    layer: int
    blocked_by_layer: int
    start: float
    end: float


@dataclass
class ScheduleResult:
    compute_windows: List[ComputeWindow]
    dp_comm_windows: List[DPCommWindow]
    tp_comm_windows: List[TPCommWindow]
    total_raw_comm: float
    total_exposed_comm: float
    total_raw_reconf: float
    total_exposed_reconf: float
    makespan: float
    total_hidden_comm: float = 0.0
    total_hidden_reconf: float = 0.0
    compute_makespan: float = 0.0
    total_tp_inter_fabric_wait: float = 0.0
    total_dp_fabric_wait: float = 0.0


@dataclass
class PPScheduleResult:
    rows: List[dict]
    total_wait_time: float = 0.0
    total_bubble_time: float = 0.0
    wait_row_count: int = 0
    bubble_row_count: int = 0
    makespan: float = 0.0


def _parameters(inputs: Any):
    return inputs.parameters if hasattr(inputs, "parameters") else inputs


def _mode_code(perf_model: Any) -> int:
    raw_mode = 0
    if perf_model is not None and hasattr(perf_model, "_param"):
        raw_mode = perf_model._param("dp_overlap_mode", 0)

    if isinstance(raw_mode, str):
        mapping = {"lump": 0, "layerwise": 1, "bucketed": 2}
        return mapping.get(raw_mode.lower(), 0)
    try:
        return int(raw_mode)
    except (TypeError, ValueError):
        return 0


def _bucket_size_bytes(perf_model: Any, layer_bytes: float) -> float:
    # Phase II uses uniform per-layer gradient sizes. A configured bucket size of
    # 0 therefore resolves to one layer's worth of gradients, and the scheduler
    # clamps any smaller value up to that same minimum.
    if perf_model is None or not hasattr(perf_model, "_param"):
        return layer_bytes

    raw_value = perf_model._param("dp_bucket_size_bytes", layer_bytes)
    try:
        bucket_size = float(raw_value)
    except (TypeError, ValueError):
        bucket_size = layer_bytes
    return max(layer_bytes, bucket_size)


def _compute_makespan(compute_windows: List[ComputeWindow]) -> float:
    return max((window.end or 0.0) for window in compute_windows) if compute_windows else 0.0


def _overlap_duration(start: Optional[float], end: Optional[float], boundary: float) -> float:
    if start is None or end is None:
        return 0.0
    return max(0.0, min(end, boundary) - start)


def _compute_intervals(compute_windows: List[ComputeWindow]) -> List[tuple[float, float]]:
    intervals: List[tuple[float, float]] = []
    for window in compute_windows:
        if window.start is None or window.end is None or window.end <= window.start:
            continue
        intervals.append((float(window.start), float(window.end)))
    intervals.sort()
    return intervals


def _trim_intervals_after(intervals: List[tuple[float, float]], start_floor: float) -> List[tuple[float, float]]:
    trimmed: List[tuple[float, float]] = []
    for start, end in intervals:
        if end <= start_floor:
            continue
        clipped_start = max(start, start_floor)
        if clipped_start < end:
            trimmed.append((clipped_start, end))
    return trimmed


def _intervals_duration(intervals: List[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _interval_overlap_duration(
    intervals: List[tuple[float, float]],
    start: Optional[float],
    end: Optional[float],
) -> float:
    if start is None or end is None or end <= start:
        return 0.0

    total = 0.0
    for interval_start, interval_end in intervals:
        overlap_start = max(start, interval_start)
        overlap_end = min(end, interval_end)
        if overlap_start < overlap_end:
            total += overlap_end - overlap_start
    return total


def _consume_interval(
    intervals: List[tuple[float, float]],
    start: Optional[float],
    end: Optional[float],
) -> tuple[List[tuple[float, float]], float]:
    if start is None or end is None or end <= start:
        return intervals[:], 0.0

    remaining: List[tuple[float, float]] = []
    consumed = 0.0

    for interval_start, interval_end in intervals:
        if interval_end <= start or interval_start >= end:
            remaining.append((interval_start, interval_end))
            continue

        overlap_start = max(start, interval_start)
        overlap_end = min(end, interval_end)
        if overlap_start < overlap_end:
            consumed += overlap_end - overlap_start

        if interval_start < overlap_start:
            remaining.append((interval_start, overlap_start))
        if overlap_end < interval_end:
            remaining.append((overlap_end, interval_end))

    return remaining, consumed


def _tp_blocking_end_overrides(
    tp_windows: List[TPCommWindow],
) -> Dict[int, float]:
    blocking_end_by_layer: Dict[int, float] = {}
    for window in tp_windows:
        if window.end is None:
            continue
        blocking_end_by_layer[window.layer] = max(
            blocking_end_by_layer.get(window.layer, 0.0),
            float(window.end),
        )
    return blocking_end_by_layer


def _compute_windows_equal(
    lhs: List[ComputeWindow],
    rhs: List[ComputeWindow],
    tolerance: float = 1e-15,
) -> bool:
    if len(lhs) != len(rhs):
        return False

    for left, right in zip(lhs, rhs):
        if (
            left.stage != right.stage
            or left.microbatch != right.microbatch
            or left.layer != right.layer
            or left.sublayer != right.sublayer
        ):
            return False
        if abs(float(left.start or 0.0) - float(right.start or 0.0)) > tolerance:
            return False
        if abs(float(left.end or 0.0) - float(right.end or 0.0)) > tolerance:
            return False
    return True


def build_backward_compute_windows(
    inputs,
    deepflow_outputs,
    tp_blocking_end_overrides: Optional[Dict[int, float]] = None,
) -> List[ComputeWindow]:
    params = _parameters(inputs)
    perf_model = deepflow_outputs
    if perf_model is None or not hasattr(perf_model, "compute_time_backward_pass"):
        raise ValueError("build_backward_compute_windows expects a PerformanceModel-like object as deepflow_outputs.")

    layers = int(params["layers"])
    parallel_divisor = float(
        params["data_parallel_degree"] * params["tensor_parallel_degree"] * params["pipeline_parallel_degree"]
    )
    parallel_divisor = max(1.0, parallel_divisor)
    layer_duration = float(perf_model.compute_time_backward_pass()) / parallel_divisor
    pre_tp_duration = BACKWARD_PRE_TP_ALPHA * layer_duration
    post_tp_duration = layer_duration - pre_tp_duration
    post_tp_pre_dp_duration = BACKWARD_POST_TP_PRE_DP_ALPHA * post_tp_duration
    post_tp_post_dp_duration = post_tp_duration - post_tp_pre_dp_duration
    bwd_tp_intra = float(perf_model.backward_tensor_model_intra())
    bwd_tp_inter = float(perf_model.backward_tensor_parallel_inter())
    tp_degree_intra = int(params["intra_node_tensor_parallel_degree"])
    tp_degree_inter = int(params["inter_node_tensor_parallel_degree"])

    windows: List[ComputeWindow] = []
    current_time = 0.0
    tp_intra_free_time = 0.0
    tp_inter_free_time = 0.0
    for layer_id in range(layers - 1, -1, -1):
        start = current_time
        pre_tp_end = start + pre_tp_duration
        dp_ready_end = pre_tp_end + post_tp_pre_dp_duration
        end = start + layer_duration

        tp_ready_time = pre_tp_end
        tp_intra_end = tp_ready_time
        if bwd_tp_intra > 0.0 and tp_degree_intra > 1:
            tp_intra_start = max(tp_ready_time, tp_intra_free_time)
            tp_intra_end = tp_intra_start + bwd_tp_intra
            tp_intra_free_time = tp_intra_end

        tp_blocking_end = tp_intra_end
        if bwd_tp_inter > 0.0 and tp_degree_inter > 1:
            tp_inter_start = max(tp_intra_end, tp_inter_free_time)
            tp_inter_end = tp_inter_start + bwd_tp_inter
            tp_inter_free_time = tp_inter_end
            tp_blocking_end = tp_inter_end

        if tp_blocking_end_overrides is not None and layer_id in tp_blocking_end_overrides:
            tp_blocking_end = max(tp_blocking_end, float(tp_blocking_end_overrides[layer_id]))

        windows.append(
            ComputeWindow(
                stage=0,
                microbatch=0,
                layer=layer_id,
                sublayer=BWD_COMP_PRE_TP,
                start=start,
                end=pre_tp_end,
            )
        )
        windows.append(
            ComputeWindow(
                stage=0,
                microbatch=0,
                layer=layer_id,
                sublayer=BWD_COMP_POST_TP_PRE_DP,
                start=pre_tp_end,
                end=dp_ready_end,
            )
        )
        windows.append(
            ComputeWindow(
                stage=0,
                microbatch=0,
                layer=layer_id,
                sublayer=BWD_COMP_POST_TP_POST_DP,
                start=dp_ready_end,
                end=end,
            )
        )
        current_time = max(end, tp_blocking_end)
    return windows


def build_overlap_schedule(inputs, perf_model, max_iterations: int = 32) -> ScheduleResult:
    compute_windows = build_backward_compute_windows(inputs, perf_model)

    for _ in range(max_iterations):
        tp_windows = build_backward_tp_comm_windows(inputs, perf_model, compute_windows)
        dp_windows = build_dp_comm_windows(inputs, perf_model, compute_windows=compute_windows)
        schedule = schedule_dp_overlap(compute_windows, dp_windows, tp_windows=tp_windows)

        delayed_tp_blocking_end = _tp_blocking_end_overrides(schedule.tp_comm_windows)
        next_compute_windows = build_backward_compute_windows(
            inputs,
            perf_model,
            tp_blocking_end_overrides=delayed_tp_blocking_end,
        )
        if _compute_windows_equal(compute_windows, next_compute_windows):
            return schedule
        compute_windows = next_compute_windows

    final_tp_windows = build_backward_tp_comm_windows(inputs, perf_model, compute_windows)
    final_dp_windows = build_dp_comm_windows(inputs, perf_model, compute_windows=compute_windows)
    return schedule_dp_overlap(compute_windows, final_dp_windows, tp_windows=final_tp_windows)


def _backward_dp_ready_windows(compute_windows: List[ComputeWindow]) -> List[ComputeWindow]:
    return [window for window in compute_windows if window.sublayer == BWD_COMP_POST_TP_PRE_DP]


def _backward_layer_phase_windows(
    compute_windows: List[ComputeWindow],
) -> List[tuple[int, ComputeWindow, ComputeWindow, ComputeWindow]]:
    by_layer: dict[int, dict[str, ComputeWindow]] = {}
    for window in compute_windows:
        layer_windows = by_layer.setdefault(window.layer, {})
        layer_windows[window.sublayer] = window

    ordered: List[tuple[int, ComputeWindow, ComputeWindow, ComputeWindow]] = []
    for layer in sorted(by_layer.keys(), reverse=True):
        pre_window = by_layer[layer][BWD_COMP_PRE_TP]
        post_pre_dp_window = by_layer[layer][BWD_COMP_POST_TP_PRE_DP]
        post_post_dp_window = by_layer[layer][BWD_COMP_POST_TP_POST_DP]
        ordered.append((layer, pre_window, post_pre_dp_window, post_post_dp_window))
    return ordered


def build_backward_tp_comm_windows(inputs, perf_model, compute_windows: List[ComputeWindow]) -> List[TPCommWindow]:
    params = _parameters(inputs)
    compute_end = _compute_makespan(compute_windows)
    tp_intra_free_time = 0.0
    tp_inter_free_time = 0.0
    tp_windows: List[TPCommWindow] = []

    bwd_tp_intra = float(perf_model.backward_tensor_model_intra())
    bwd_tp_inter = float(perf_model.backward_tensor_parallel_inter())
    tp_degree_intra = int(params["intra_node_tensor_parallel_degree"])
    tp_degree_inter = int(params["inter_node_tensor_parallel_degree"])

    if bwd_tp_intra <= 0.0 and bwd_tp_inter <= 0.0:
        return tp_windows

    for layer, pre_window, _post_pre_dp_window, _post_post_dp_window in _backward_layer_phase_windows(compute_windows):
        tp_ready_time = float(pre_window.end or 0.0)

        tp_intra_end = tp_ready_time
        if bwd_tp_intra > 0.0 and tp_degree_intra > 1:
            tp_intra_start = max(tp_ready_time, tp_intra_free_time)
            tp_intra_end = tp_intra_start + bwd_tp_intra
            hidden_intra = _overlap_duration(tp_intra_start, tp_intra_end, compute_end)
            tp_windows.append(
                TPCommWindow(
                    layer=layer,
                    locality="INTRA",
                    ready_time=tp_ready_time,
                    raw_comm_duration=bwd_tp_intra,
                    start=tp_intra_start,
                    end=tp_intra_end,
                    hidden_comm_duration=min(bwd_tp_intra, hidden_intra),
                    exposed_comm_duration=bwd_tp_intra - min(bwd_tp_intra, hidden_intra),
                )
            )
            tp_intra_free_time = tp_intra_end

        tp_inter_ready = tp_intra_end
        if bwd_tp_inter > 0.0 and tp_degree_inter > 1:
            tp_inter_start = max(tp_inter_ready, tp_inter_free_time)
            tp_inter_end = tp_inter_start + bwd_tp_inter
            hidden_inter = _overlap_duration(tp_inter_start, tp_inter_end, compute_end)
            tp_windows.append(
                TPCommWindow(
                    layer=layer,
                    locality="INTER",
                    ready_time=tp_inter_ready,
                    raw_comm_duration=bwd_tp_inter,
                    start=tp_inter_start,
                    end=tp_inter_end,
                    hidden_comm_duration=min(bwd_tp_inter, hidden_inter),
                    exposed_comm_duration=bwd_tp_inter - min(bwd_tp_inter, hidden_inter),
                )
            )
            tp_inter_free_time = tp_inter_end

    return tp_windows


def build_backward_wait_windows(compute_windows: List[ComputeWindow]) -> List[WaitWindow]:
    wait_windows: List[WaitWindow] = []
    layer_windows = _backward_layer_phase_windows(compute_windows)
    for idx in range(len(layer_windows) - 1):
        current_layer, _current_pre, _current_post_pre_dp, current_post_post_dp = layer_windows[idx]
        next_layer, next_pre, _next_post_pre_dp, _next_post_post_dp = layer_windows[idx + 1]
        natural_next_start = float(current_post_post_dp.end or 0.0)
        actual_next_start = float(next_pre.start or 0.0)
        if actual_next_start > natural_next_start:
            wait_windows.append(
                WaitWindow(
                    layer=next_layer,
                    blocked_by_layer=current_layer,
                    start=natural_next_start,
                    end=actual_next_start,
                )
            )
    return wait_windows


def build_dp_comm_windows(inputs, perf_model, compute_windows: Optional[List[ComputeWindow]] = None) -> List[DPCommWindow]:
    params = _parameters(inputs)
    if compute_windows is None:
        compute_windows = build_backward_compute_windows(inputs, perf_model)
    dp_ready_windows = _backward_dp_ready_windows(compute_windows)
    if perf_model is None or not hasattr(perf_model, "dp_gradient_bytes_per_layer"):
        raise ValueError("build_dp_comm_windows expects a PerformanceModel with DP overlap helpers.")

    mode = _mode_code(perf_model)
    layer_bytes = float(perf_model.dp_gradient_bytes_per_layer())
    if layer_bytes <= 0.0:
        return []

    windows: List[DPCommWindow] = []
    layers = int(params["layers"])
    dp_intra_degree = int(params["intra_node_data_parallel_degree"])
    dp_inter_degree = int(params["inter_node_data_parallel_degree"])
    dp_locality = "INTRA+INTER" if dp_intra_degree > 1 and dp_inter_degree > 1 else ("INTER" if dp_inter_degree > 1 else "INTRA")

    if mode == 0:
        total_bytes = layer_bytes * layers
        raw_comm = layers * float(perf_model.dp_allreduce_raw_for_bytes(layer_bytes))
        raw_reconf = layers * float(perf_model.dp_inter_reconfiguration_time_raw_per_collective())
        windows.append(
            DPCommWindow(
                bucket_id=0,
                stage=0,
                bytes=total_bytes,
                ready_time=_compute_makespan(compute_windows),
                raw_comm_duration=raw_comm,
                raw_reconf_duration=raw_reconf,
                unit_kind="lump",
                allow_pre_ready_reconfiguration=False,
                first_layer=0,
                last_layer=max(0, layers - 1),
                layer_count=layers,
                topology_signature=_topology_signature(
                    parallelism="DP",
                    locality=dp_locality,
                    unit_kind="lump",
                ),
                configured_reconf_duration=raw_reconf,
            )
        )
        return windows

    if mode == 1:
        raw_comm = float(perf_model.dp_allreduce_raw_for_bytes(layer_bytes))
        raw_reconf = float(perf_model.dp_inter_reconfiguration_time_raw_per_collective())
        topology_signature = _topology_signature(
            parallelism="DP",
            locality=dp_locality,
            unit_kind="layerwise",
        )
        for bucket_id, window in enumerate(dp_ready_windows):
            windows.append(
                DPCommWindow(
                    bucket_id=bucket_id,
                    stage=0,
                    bytes=layer_bytes,
                    ready_time=float(window.end or 0.0),
                    raw_comm_duration=raw_comm,
                    raw_reconf_duration=raw_reconf,
                    unit_kind="layerwise",
                    first_layer=window.layer,
                    last_layer=window.layer,
                    layer_count=1,
                    topology_signature=topology_signature,
                    configured_reconf_duration=raw_reconf,
                )
            )
        return windows

    bucket_size_bytes = _bucket_size_bytes(perf_model, layer_bytes)
    current_bucket: List[ComputeWindow] = []
    current_bytes = 0.0

    for window in dp_ready_windows:
        current_bucket.append(window)
        current_bytes += layer_bytes
        if current_bytes + 1e-18 >= bucket_size_bytes:
            raw_comm = float(perf_model.dp_allreduce_raw_for_bytes(current_bytes))
            raw_reconf = float(perf_model.dp_inter_reconfiguration_time_raw_per_collective())
            topology_signature = _topology_signature(
                parallelism="DP",
                locality=dp_locality,
                unit_kind="bucketed",
            )
            windows.append(
                DPCommWindow(
                    bucket_id=len(windows),
                    stage=0,
                    bytes=current_bytes,
                    ready_time=float(current_bucket[-1].end or 0.0),
                    raw_comm_duration=raw_comm,
                    raw_reconf_duration=raw_reconf,
                    unit_kind="bucketed",
                    first_layer=current_bucket[-1].layer,
                    last_layer=current_bucket[0].layer,
                    layer_count=len(current_bucket),
                    topology_signature=topology_signature,
                    configured_reconf_duration=raw_reconf,
                )
            )
            current_bucket = []
            current_bytes = 0.0

    if current_bucket:
        raw_comm = float(perf_model.dp_allreduce_raw_for_bytes(current_bytes))
        raw_reconf = float(perf_model.dp_inter_reconfiguration_time_raw_per_collective())
        topology_signature = _topology_signature(
            parallelism="DP",
            locality=dp_locality,
            unit_kind="bucketed",
        )
        windows.append(
            DPCommWindow(
                bucket_id=len(windows),
                stage=0,
                bytes=current_bytes,
                ready_time=float(current_bucket[-1].end or 0.0),
                raw_comm_duration=raw_comm,
                raw_reconf_duration=raw_reconf,
                unit_kind="bucketed",
                first_layer=current_bucket[-1].layer,
                last_layer=current_bucket[0].layer,
                layer_count=len(current_bucket),
                topology_signature=topology_signature,
                configured_reconf_duration=raw_reconf,
            )
        )

    return windows


def _prepare_dp_window_for_service(
    window: DPCommWindow,
    *,
    queue_head_time: float,
    current_dp_topology: Optional[str],
    compute_end: float,
    available_comm_slack_intervals: List[tuple[float, float]],
) -> List[tuple[float, float]]:
    if window.unit_kind == "lump":
        effective_reconf_duration = window.raw_reconf_duration
    else:
        if current_dp_topology is None or window.topology_signature != current_dp_topology:
            effective_reconf_duration = window.configured_reconf_duration
        else:
            effective_reconf_duration = 0.0

    window.raw_reconf_duration = effective_reconf_duration

    if effective_reconf_duration > 0.0:
        reconf_start = max(window.ready_time, queue_head_time)
        reconf_end = reconf_start + window.raw_reconf_duration
        request_time = reconf_end
    else:
        reconf_start = None
        reconf_end = None
        request_time = max(window.ready_time, queue_head_time)

    window.reconf_start = reconf_start
    window.reconf_end = reconf_end
    window.request_time = request_time

    hidden_reconf = _overlap_duration(window.reconf_start, window.reconf_end, compute_end)
    window.hidden_reconf_duration = min(window.raw_reconf_duration, hidden_reconf)
    window.exposed_reconf_duration = window.raw_reconf_duration - window.hidden_reconf_duration

    if window.hidden_reconf_duration > 0.0:
        available_comm_slack_intervals, _ = _consume_interval(
            available_comm_slack_intervals,
            window.reconf_start,
            min(float(window.reconf_end or 0.0), compute_end),
        )

    return available_comm_slack_intervals


def schedule_dp_overlap(
    compute_windows: List[ComputeWindow],
    dp_windows: List[DPCommWindow],
    tp_windows: Optional[List[TPCommWindow]] = None,
) -> ScheduleResult:
    compute_end = _compute_makespan(compute_windows)
    available_comm_slack_intervals = _compute_intervals(compute_windows)
    tp_windows = tp_windows or []
    fabric_free_time = 0.0
    dp_lane_free_time = 0.0
    current_dp_topology: Optional[str] = None
    total_tp_inter_fabric_wait = 0.0
    total_dp_fabric_wait = 0.0

    tp_inter_windows = [window for window in tp_windows if window.locality == "INTER"]
    for window in tp_inter_windows:
        window.request_time = window.ready_time
        window.fabric_wait_duration = 0.0
    tp_inter_windows.sort(key=lambda window: float(window.ready_time))

    dp_index = 0
    tp_index = 0
    next_dp_window: Optional[DPCommWindow] = None

    if dp_windows:
        next_dp_window = dp_windows[0]
        available_comm_slack_intervals = _prepare_dp_window_for_service(
            next_dp_window,
            queue_head_time=dp_lane_free_time,
            current_dp_topology=current_dp_topology,
            compute_end=compute_end,
            available_comm_slack_intervals=available_comm_slack_intervals,
        )

    while tp_index < len(tp_inter_windows) or next_dp_window is not None:
        tp_request = float(tp_inter_windows[tp_index].request_time or 0.0) if tp_index < len(tp_inter_windows) else float("inf")
        dp_request = float(next_dp_window.request_time or 0.0) if next_dp_window is not None else float("inf")

        choose_tp = False
        if tp_index < len(tp_inter_windows):
            if next_dp_window is None or tp_request < dp_request or tp_request == dp_request:
                choose_tp = True

        if choose_tp:
            tp_window = tp_inter_windows[tp_index]
            tp_window.start = max(tp_request, fabric_free_time)
            tp_window.end = tp_window.start + tp_window.raw_comm_duration
            tp_window.fabric_wait_duration = tp_window.start - tp_request
            hidden_inter = _overlap_duration(tp_window.start, tp_window.end, compute_end)
            tp_window.hidden_comm_duration = min(tp_window.raw_comm_duration, hidden_inter)
            tp_window.exposed_comm_duration = tp_window.raw_comm_duration - tp_window.hidden_comm_duration
            total_tp_inter_fabric_wait += tp_window.fabric_wait_duration
            fabric_free_time = tp_window.end
            tp_index += 1
            continue

        window = next_dp_window
        assert window is not None

        request_time = float(window.request_time or 0.0)
        comm_start = max(request_time, fabric_free_time)
        comm_end = comm_start + window.raw_comm_duration
        window.comm_start = comm_start
        window.comm_end = comm_end
        window.start = window.reconf_start if window.raw_reconf_duration > 0 else comm_start
        window.end = comm_end
        window.fabric_wait_duration = comm_start - request_time
        total_dp_fabric_wait += window.fabric_wait_duration

        local_comm_floor = float(window.reconf_start if window.reconf_start is not None else request_time)
        local_comm_slack_intervals = _trim_intervals_after(available_comm_slack_intervals, local_comm_floor)
        window.available_local_comm_slack = _intervals_duration(local_comm_slack_intervals)

        hidden_comm = _interval_overlap_duration(
            local_comm_slack_intervals,
            window.comm_start,
            min(float(window.comm_end or 0.0), compute_end),
        )
        window.hidden_comm_duration = min(window.raw_comm_duration, hidden_comm)
        window.exposed_comm_duration = window.raw_comm_duration - window.hidden_comm_duration

        if window.hidden_comm_duration > 0.0:
            available_comm_slack_intervals, _ = _consume_interval(
                available_comm_slack_intervals,
                window.comm_start,
                min(float(window.comm_end or 0.0), compute_end),
            )

        fabric_free_time = comm_end
        dp_lane_free_time = comm_end
        if window.topology_signature != "dp_none":
            current_dp_topology = window.topology_signature

        dp_index += 1
        if dp_index < len(dp_windows):
            next_dp_window = dp_windows[dp_index]
            available_comm_slack_intervals = _prepare_dp_window_for_service(
                next_dp_window,
                queue_head_time=dp_lane_free_time,
                current_dp_topology=current_dp_topology,
                compute_end=compute_end,
                available_comm_slack_intervals=available_comm_slack_intervals,
            )
        else:
            next_dp_window = None

    total_raw_comm = sum(window.raw_comm_duration for window in dp_windows)
    total_exposed_comm = sum(window.exposed_comm_duration for window in dp_windows)
    total_raw_reconf = sum(window.raw_reconf_duration for window in dp_windows)
    total_exposed_reconf = sum(window.exposed_reconf_duration for window in dp_windows)
    makespan = max(
        compute_end,
        max((window.end or 0.0) for window in dp_windows) if dp_windows else 0.0,
        max((window.end or 0.0) for window in tp_windows) if tp_windows else 0.0,
    )

    return ScheduleResult(
        compute_windows=compute_windows,
        dp_comm_windows=dp_windows,
        tp_comm_windows=tp_windows,
        total_raw_comm=total_raw_comm,
        total_exposed_comm=total_exposed_comm,
        total_raw_reconf=total_raw_reconf,
        total_exposed_reconf=total_exposed_reconf,
        makespan=makespan,
        total_hidden_comm=total_raw_comm - total_exposed_comm,
        total_hidden_reconf=total_raw_reconf - total_exposed_reconf,
        compute_makespan=compute_end,
        total_tp_inter_fabric_wait=total_tp_inter_fabric_wait,
        total_dp_fabric_wait=total_dp_fabric_wait,
    )


def summarize(result: ScheduleResult) -> dict:
    return {
        "unit_count": len(result.dp_comm_windows),
        "compute_makespan": result.compute_makespan,
        "schedule_makespan": result.makespan,
        "total_raw_comm": result.total_raw_comm,
        "total_hidden_comm": result.total_hidden_comm,
        "total_exposed_comm": result.total_exposed_comm,
        "total_raw_reconf": result.total_raw_reconf,
        "total_hidden_reconf": result.total_hidden_reconf,
        "total_exposed_reconf": result.total_exposed_reconf,
    }


def _candidate_signature(
    collective_type: str,
    parallelism: str,
    locality: str,
    degree: Any,
    bytes_to_transfer: Any,
) -> str:
    return (
        f"collective={collective_type}|parallelism={parallelism}|locality={locality}|"
        f"degree={degree}|bytes={bytes_to_transfer}"
    )


def _topology_signature(
    *,
    parallelism: str,
    locality: str,
    unit_kind: str = "NONE",
) -> str:
    if parallelism == "TP":
        return "tp_intra" if locality == "INTRA" else "tp_inter"
    if parallelism == "PP":
        return "pp_intra" if locality == "INTRA" else "pp_inter"
    if parallelism == "DP":
        locality_key = {
            "INTRA": "intra",
            "INTER": "inter",
            "INTRA+INTER": "hybrid",
        }.get(locality, locality.lower())
        unit_key = unit_kind.lower() if unit_kind not in ("NONE", "", None) else "generic"
        return f"dp_{locality_key}_{unit_key}"
    return "none"


def _event_provenance(
    *,
    scheduler_explicit: str,
    ordering_source: str,
) -> str:
    if scheduler_explicit == "yes" and ordering_source in {
        "phase2_dp_scheduler",
        "phase3_bwd_tp_scheduler",
        "phase4a_shared_fabric",
        "phase5a_pp_scheduler",
        "phase5b_pp_idle_scheduler",
    }:
        return "scheduler_explicit"
    if scheduler_explicit == "yes":
        return "scheduler_derived"
    if ordering_source.startswith("anchored_"):
        return "scaffold_anchored"
    if ordering_source == "phase_scaffold_boundary":
        return "scaffold_boundary"
    if ordering_source == "phase_scaffold_serial":
        return "scaffold_serial"
    return "scaffold_other"


def _add_csv_event(
    rows: List[dict],
    *,
    event: str,
    event_type: str,
    start: float,
    end: float,
    bytes_to_transfer: Any = "NONE",
    collective_type: str = "NONE",
    parallelism: str = "NONE",
    locality: str = "NONE",
    degree: Any = 0,
    lane: str = "MAIN",
    step: int = 0,
    layer: Any = "NONE",
    first_layer: Any = "NONE",
    last_layer: Any = "NONE",
    layer_count: Any = 0,
    bucket_id: Any = "NONE",
    unit_kind: str = "NONE",
    ready_time: Any = "NONE",
    raw_comm_duration: Any = "NONE",
    raw_reconf_duration: Any = "NONE",
    hidden_comm_duration: Any = "NONE",
    exposed_comm_duration: Any = "NONE",
    hidden_reconf_duration: Any = "NONE",
    exposed_reconf_duration: Any = "NONE",
    stage_index: Any = "NONE",
    microbatch_index: Any = "NONE",
    scheduler_explicit: str = "no",
    ordering_source: str = "model_scaffold",
    topology_candidate_tag: Optional[str] = None,
    topology_signature: Optional[str] = None,
) -> None:
    duration = max(0.0, float(end) - float(start))
    if topology_candidate_tag is None:
        topology_candidate_tag = _candidate_signature(
            collective_type,
            parallelism,
            locality,
            degree,
            bytes_to_transfer,
        )
    if topology_signature is None:
        topology_signature = _topology_signature(
            parallelism=parallelism,
            locality=locality,
            unit_kind=unit_kind,
        )
    event_provenance = _event_provenance(
        scheduler_explicit=scheduler_explicit,
        ordering_source=ordering_source,
    )

    rows.append(
        {
            "Event": event,
            "Type": event_type,
            "start time": start,
            "end time": end,
            "duration": duration,
            "Bytes to be transferred": bytes_to_transfer,
            "Collective type": collective_type,
            "Parallelism": parallelism,
            "Locality": locality,
            "Degree": degree,
            "Lane": lane,
            "Step": step,
            "Layer index": layer,
            "First layer": first_layer,
            "Last layer": last_layer,
            "Layer count": layer_count,
            "Bucket id": bucket_id,
            "Unit kind": unit_kind,
            "Ready time": ready_time,
            "Raw comm duration": raw_comm_duration,
            "Raw reconf duration": raw_reconf_duration,
            "Hidden comm duration": hidden_comm_duration,
            "Exposed comm duration": exposed_comm_duration,
            "Hidden reconf duration": hidden_reconf_duration,
            "Exposed reconf duration": exposed_reconf_duration,
            "Stage index": stage_index,
            "Microbatch index": microbatch_index,
            "Scheduler explicit": scheduler_explicit,
            "Event provenance": event_provenance,
            "Ordering source": ordering_source,
            "topology_signature": topology_signature,
            "Topology candidate tag": topology_candidate_tag,
        }
    )


def _annotate_previous_event_fields(rows: List[dict]) -> None:
    previous_row: Optional[dict] = None
    for row in rows:
        if previous_row is None:
            row["previous_event_type"] = "NONE"
            row["previous_topology_signature"] = "none"
            row["gap_from_previous_s"] = "NONE"
        else:
            row["previous_event_type"] = previous_row["Event"]
            row["previous_topology_signature"] = previous_row["topology_signature"]
            row["gap_from_previous_s"] = float(row["start time"]) - float(previous_row["end time"])
        previous_row = row


def _pipeline_stage_spans(layers: int, pipeline_parallel_degree: int) -> List[tuple[int, int, int, int]]:
    stage_count = max(1, pipeline_parallel_degree)
    base_layers = layers // stage_count
    remainder = layers % stage_count

    spans: List[tuple[int, int, int, int]] = []
    first_layer = 0
    for stage in range(stage_count):
        layer_count = base_layers + (1 if stage < remainder else 0)
        last_layer = first_layer + layer_count - 1
        spans.append((stage, first_layer, max(first_layer, last_layer), layer_count))
        first_layer = last_layer + 1
    return spans


def _pp_link_metadata(
    *,
    intra_duration: float,
    inter_duration: float,
    intra_degree: int,
    inter_degree: int,
) -> tuple[float, str, int]:
    if inter_duration >= intra_duration:
        return inter_duration, "INTER", inter_degree
    return intra_duration, "INTRA", intra_degree


def _forward_local_duration(
    params: Dict[str, Any],
    perf_model: Any,
) -> float:
    parallel_divisor = max(
        1.0,
        float(
            params["data_parallel_degree"]
            * params["tensor_parallel_degree"]
            * params["pipeline_parallel_degree"]
        ),
    )
    fwd_compute = float(perf_model.compute_time_forward_pass()) / parallel_divisor
    fwd_tp_intra = float(perf_model.forward_tensor_model_intra())
    fwd_tp_inter = float(perf_model.forward_tensor_parallel_inter())
    per_layer_duration = fwd_compute + fwd_tp_intra + fwd_tp_inter
    return int(params["layers"]) * per_layer_duration


def _backward_local_duration(
    schedule: ScheduleResult,
) -> float:
    tp_lane_end = max((float(window.end or 0.0) for window in schedule.tp_comm_windows), default=0.0)
    return max(schedule.compute_makespan, tp_lane_end)


def _record_pp_idle_row(
    result: PPScheduleResult,
    *,
    event: str,
    start: float,
    end: float,
    stage: int,
    microbatch: int,
    first_layer: int,
    last_layer: int,
    layer_count: int,
    lane: str,
    topology_candidate_tag: str,
) -> None:
    if end <= start:
        return

    _add_csv_event(
        result.rows,
        event=event,
        event_type="Bubble" if event.endswith("BUBBLE") else "Wait",
        start=start,
        end=end,
        bytes_to_transfer="NONE",
        collective_type="NONE",
        parallelism="PP",
        locality="NONE",
        degree=0,
        lane=lane,
        first_layer=first_layer,
        last_layer=last_layer,
        layer_count=layer_count,
        stage_index=stage,
        microbatch_index=microbatch,
        scheduler_explicit="yes",
        ordering_source="phase5b_pp_idle_scheduler",
        topology_candidate_tag=topology_candidate_tag,
        topology_signature="none",
    )
    duration = float(end) - float(start)
    if event.endswith("BUBBLE"):
        result.total_bubble_time += duration
        result.bubble_row_count += 1
    else:
        result.total_wait_time += duration
        result.wait_row_count += 1


def build_pp_schedule_result(
    inputs: Any,
    perf_model: Any,
    *,
    schedule: Optional[ScheduleResult] = None,
) -> PPScheduleResult:
    params = _parameters(inputs)
    pipeline_parallel_degree = int(params["pipeline_parallel_degree"])
    microbatch_count = int(params["number_of_microbatches_per_minibatch"])
    result = PPScheduleResult(rows=[])
    if pipeline_parallel_degree <= 1 or microbatch_count <= 0:
        return result

    if schedule is None:
        schedule = build_overlap_schedule(inputs, perf_model)

    forward_local_duration = _forward_local_duration(params, perf_model)
    backward_local_duration = _backward_local_duration(schedule)
    activation_volume = (
        float(params["microbatch_size"])
        * float(params["hidden_layer_dimension_for_attention_sublayers"])
        * float(params["attention_heads"])
        * float(params["context"])
        * float(params["activation_precision"])
        / 8.0
    )
    error_volume = (
        float(params["microbatch_size"])
        * float(params["context"])
        * float(params["hidden_layer_dimension_for_attention_sublayers"])
        * float(params["attention_heads"])
        * float(params["gradient_precision"])
        / 8.0
    )
    fwd_pp_intra = float(perf_model.forward_pipeline_parallel_intra())
    fwd_pp_inter = float(perf_model.forward_pipeline_parallel_inter())
    bwd_pp_intra = float(perf_model.backward_pipeline_parallel_intra())
    bwd_pp_inter = float(perf_model.backward_pipeline_parallel_inter())

    stage_spans = _pipeline_stage_spans(int(params["layers"]), pipeline_parallel_degree)
    fwd_pp_duration, fwd_pp_locality, fwd_pp_degree = _pp_link_metadata(
        intra_duration=fwd_pp_intra,
        inter_duration=fwd_pp_inter,
        intra_degree=int(params["intra_node_pipeline_parallel_degree"]),
        inter_degree=int(params["inter_node_pipeline_parallel_degree"]),
    )
    bwd_pp_duration, bwd_pp_locality, bwd_pp_degree = _pp_link_metadata(
        intra_duration=bwd_pp_intra,
        inter_duration=bwd_pp_inter,
        intra_degree=int(params["intra_node_pipeline_parallel_degree"]),
        inter_degree=int(params["inter_node_pipeline_parallel_degree"]),
    )

    ordering_source = "phase5a_pp_scheduler"
    stage_idle_lane = lambda stage: f"PP_STAGE_{stage}_IDLE"
    stage_queue_lane = lambda stage: f"PP_STAGE_{stage}_QUEUE"
    stage_compute_free: List[float] = [0.0 for _ in range(pipeline_parallel_degree)]
    forward_ready: List[List[float]] = [
        [0.0 for _ in range(microbatch_count)] for _ in range(pipeline_parallel_degree)
    ]
    forward_recv_start: List[List[Optional[float]]] = [
        [None for _ in range(microbatch_count)] for _ in range(pipeline_parallel_degree)
    ]
    forward_done: List[List[float]] = [
        [0.0 for _ in range(microbatch_count)] for _ in range(pipeline_parallel_degree)
    ]
    forward_link_free: List[float] = [0.0 for _ in range(max(0, pipeline_parallel_degree - 1))]
    stage_last_useful_end: List[float] = [0.0 for _ in range(pipeline_parallel_degree)]
    stage_backward_started: List[bool] = [False for _ in range(pipeline_parallel_degree)]

    # Explicitize the frozen model as a coarse GPipe-style fill-drain schedule:
    # all forward microbatches first, then all backward microbatches.
    for microbatch in range(microbatch_count):
        for stage, first_layer, last_layer, layer_count in stage_spans:
            stage_free = stage_compute_free[stage]
            recv_ready = forward_ready[stage][microbatch]
            recv_start = forward_recv_start[stage][microbatch]

            if stage > 0:
                if microbatch == 0:
                    bubble_end = recv_start if recv_start is not None else recv_ready
                    _record_pp_idle_row(
                        result,
                        event="FWD_PP_BUBBLE",
                        start=stage_free,
                        end=bubble_end,
                        stage=stage,
                        microbatch=microbatch,
                        first_layer=first_layer,
                        last_layer=last_layer,
                        layer_count=layer_count,
                        lane=stage_idle_lane(stage),
                        topology_candidate_tag=(
                            f"stage={stage}|microbatch={microbatch}|phase=fwd|reason=startup_bubble"
                        ),
                    )

                wait_for_recv_start = max(stage_free, recv_start if recv_start is not None else stage_free)
                if wait_for_recv_start < recv_ready:
                    _record_pp_idle_row(
                        result,
                        event="FWD_PP_WAIT_FOR_RECV",
                        start=wait_for_recv_start,
                        end=recv_ready,
                        stage=stage,
                        microbatch=microbatch,
                        first_layer=first_layer,
                        last_layer=last_layer,
                        layer_count=layer_count,
                        lane=stage_idle_lane(stage),
                        topology_candidate_tag=(
                            f"stage={stage}|microbatch={microbatch}|phase=fwd|"
                            f"reason=waiting_for_upstream_activation_completion"
                        ),
                    )

                if recv_ready < stage_free:
                    _record_pp_idle_row(
                        result,
                        event="FWD_PP_WAIT_FOR_STAGE",
                        start=recv_ready,
                        end=stage_free,
                        stage=stage,
                        microbatch=microbatch,
                        first_layer=first_layer,
                        last_layer=last_layer,
                        layer_count=layer_count,
                        lane=stage_queue_lane(stage),
                        topology_candidate_tag=(
                            f"stage={stage}|microbatch={microbatch}|phase=fwd|"
                            f"reason=stage_busy_after_activation_ready"
                        ),
                    )

            compute_start = max(stage_free, recv_ready)
            compute_end = compute_start + forward_local_duration
            _add_csv_event(
                result.rows,
                event="FWD_PP_STAGE_COMPUTE",
                event_type="Compute",
                start=compute_start,
                end=compute_end,
                lane=f"PP_STAGE_{stage}",
                first_layer=first_layer,
                last_layer=last_layer,
                layer_count=layer_count,
                stage_index=stage,
                microbatch_index=microbatch,
                scheduler_explicit="yes",
                ordering_source=ordering_source,
                topology_candidate_tag=f"stage={stage}|microbatch={microbatch}|phase=fwd_compute",
                topology_signature="none",
            )
            stage_compute_free[stage] = compute_end
            forward_done[stage][microbatch] = compute_end
            stage_last_useful_end[stage] = max(stage_last_useful_end[stage], compute_end)

            if stage >= pipeline_parallel_degree - 1 or fwd_pp_duration <= 0.0:
                continue

            send_start = max(compute_end, forward_link_free[stage])
            send_end = send_start + fwd_pp_duration
            topology_tag = (
                f"from_stage={stage}|to_stage={stage + 1}|microbatch={microbatch}|"
                f"phase=fwd|direction=send_recv"
            )
            _add_csv_event(
                result.rows,
                event="FWD_PP_SEND",
                event_type="Comm",
                start=send_start,
                end=send_end,
                bytes_to_transfer=activation_volume,
                collective_type="P2P",
                parallelism="PP",
                locality=fwd_pp_locality,
                degree=fwd_pp_degree,
                lane=f"PP_EDGE_{stage}_{stage + 1}_FWD",
                first_layer=first_layer,
                last_layer=last_layer,
                layer_count=layer_count,
                ready_time=compute_end,
                raw_comm_duration=fwd_pp_duration,
                stage_index=stage,
                microbatch_index=microbatch,
                scheduler_explicit="yes",
                ordering_source=ordering_source,
                topology_candidate_tag=topology_tag,
            )
            _add_csv_event(
                result.rows,
                event="FWD_PP_RECV",
                event_type="Comm",
                start=send_start,
                end=send_end,
                bytes_to_transfer=activation_volume,
                collective_type="P2P",
                parallelism="PP",
                locality=fwd_pp_locality,
                degree=fwd_pp_degree,
                lane=f"PP_EDGE_{stage}_{stage + 1}_FWD",
                first_layer=stage_spans[stage + 1][1],
                last_layer=stage_spans[stage + 1][2],
                layer_count=stage_spans[stage + 1][3],
                ready_time=send_start,
                stage_index=stage + 1,
                microbatch_index=microbatch,
                scheduler_explicit="yes",
                ordering_source=ordering_source,
                topology_candidate_tag=topology_tag,
            )
            forward_link_free[stage] = send_end
            forward_ready[stage + 1][microbatch] = send_end
            forward_recv_start[stage + 1][microbatch] = send_start
            stage_last_useful_end[stage] = max(stage_last_useful_end[stage], send_end)
            stage_last_useful_end[stage + 1] = max(stage_last_useful_end[stage + 1], send_end)

    backward_ready: List[List[float]] = [
        [0.0 for _ in range(microbatch_count)] for _ in range(pipeline_parallel_degree)
    ]
    backward_recv_start: List[List[Optional[float]]] = [
        [None for _ in range(microbatch_count)] for _ in range(pipeline_parallel_degree)
    ]
    backward_link_free: List[float] = forward_link_free[:]
    backward_stage_free: List[float] = stage_compute_free[:]

    for microbatch in range(microbatch_count - 1, -1, -1):
        for reverse_idx, (stage, first_layer, last_layer, layer_count) in enumerate(reversed(stage_spans)):
            stage = stage_spans[pipeline_parallel_degree - 1 - reverse_idx][0]
            first_layer, last_layer, layer_count = stage_spans[pipeline_parallel_degree - 1 - reverse_idx][1:]
            stage_free = backward_stage_free[stage]
            recv_ready = backward_ready[stage][microbatch]
            recv_start = backward_recv_start[stage][microbatch]
            compute_ready = max(forward_done[stage][microbatch], recv_ready)

            if stage < pipeline_parallel_degree - 1:
                if not stage_backward_started[stage]:
                    bubble_end = recv_start if recv_start is not None else recv_ready
                    _record_pp_idle_row(
                        result,
                        event="BWD_PP_BUBBLE",
                        start=stage_free,
                        end=bubble_end,
                        stage=stage,
                        microbatch=microbatch,
                        first_layer=first_layer,
                        last_layer=last_layer,
                        layer_count=layer_count,
                        lane=stage_idle_lane(stage),
                        topology_candidate_tag=(
                            f"stage={stage}|microbatch={microbatch}|phase=bwd|reason=drain_turnaround_bubble"
                        ),
                    )

                wait_for_recv_start = max(stage_free, recv_start if recv_start is not None else stage_free)
                if wait_for_recv_start < recv_ready:
                    _record_pp_idle_row(
                        result,
                        event="BWD_PP_WAIT_FOR_RECV",
                        start=wait_for_recv_start,
                        end=recv_ready,
                        stage=stage,
                        microbatch=microbatch,
                        first_layer=first_layer,
                        last_layer=last_layer,
                        layer_count=layer_count,
                        lane=stage_idle_lane(stage),
                        topology_candidate_tag=(
                            f"stage={stage}|microbatch={microbatch}|phase=bwd|"
                            f"reason=waiting_for_downstream_gradient_completion"
                        ),
                    )

                if compute_ready < stage_free:
                    _record_pp_idle_row(
                        result,
                        event="BWD_PP_WAIT_FOR_STAGE",
                        start=compute_ready,
                        end=stage_free,
                        stage=stage,
                        microbatch=microbatch,
                        first_layer=first_layer,
                        last_layer=last_layer,
                        layer_count=layer_count,
                        lane=stage_queue_lane(stage),
                        topology_candidate_tag=(
                            f"stage={stage}|microbatch={microbatch}|phase=bwd|"
                            f"reason=stage_busy_after_gradient_ready"
                        ),
                    )

            compute_start = max(stage_free, compute_ready)
            compute_end = compute_start + backward_local_duration
            _add_csv_event(
                result.rows,
                event="BWD_PP_STAGE_COMPUTE",
                event_type="Compute",
                start=compute_start,
                end=compute_end,
                lane=f"PP_STAGE_{stage}",
                first_layer=first_layer,
                last_layer=last_layer,
                layer_count=layer_count,
                stage_index=stage,
                microbatch_index=microbatch,
                scheduler_explicit="yes",
                ordering_source=ordering_source,
                topology_candidate_tag=f"stage={stage}|microbatch={microbatch}|phase=bwd_compute",
                topology_signature="none",
            )
            backward_stage_free[stage] = compute_end
            stage_backward_started[stage] = True
            stage_last_useful_end[stage] = max(stage_last_useful_end[stage], compute_end)

            if stage <= 0 or bwd_pp_duration <= 0.0:
                continue

            boundary = stage - 1
            send_start = max(compute_end, backward_link_free[boundary])
            send_end = send_start + bwd_pp_duration
            topology_tag = (
                f"from_stage={stage}|to_stage={stage - 1}|microbatch={microbatch}|"
                f"phase=bwd|direction=send_recv"
            )
            _add_csv_event(
                result.rows,
                event="BWD_PP_SEND",
                event_type="Comm",
                start=send_start,
                end=send_end,
                bytes_to_transfer=error_volume,
                collective_type="P2P",
                parallelism="PP",
                locality=bwd_pp_locality,
                degree=bwd_pp_degree,
                lane=f"PP_EDGE_{stage - 1}_{stage}_BWD",
                first_layer=first_layer,
                last_layer=last_layer,
                layer_count=layer_count,
                ready_time=compute_end,
                stage_index=stage,
                microbatch_index=microbatch,
                scheduler_explicit="yes",
                ordering_source=ordering_source,
                topology_candidate_tag=topology_tag,
            )
            _add_csv_event(
                result.rows,
                event="BWD_PP_RECV",
                event_type="Comm",
                start=send_start,
                end=send_end,
                bytes_to_transfer=error_volume,
                collective_type="P2P",
                parallelism="PP",
                locality=bwd_pp_locality,
                degree=bwd_pp_degree,
                lane=f"PP_EDGE_{stage - 1}_{stage}_BWD",
                first_layer=stage_spans[stage - 1][1],
                last_layer=stage_spans[stage - 1][2],
                layer_count=stage_spans[stage - 1][3],
                ready_time=send_start,
                stage_index=stage - 1,
                microbatch_index=microbatch,
                scheduler_explicit="yes",
                ordering_source=ordering_source,
                topology_candidate_tag=topology_tag,
            )
            backward_link_free[boundary] = send_end
            backward_ready[stage - 1][microbatch] = send_end
            backward_recv_start[stage - 1][microbatch] = send_start
            stage_last_useful_end[stage] = max(stage_last_useful_end[stage], send_end)
            stage_last_useful_end[stage - 1] = max(stage_last_useful_end[stage - 1], send_end)

    useful_makespan = max(stage_last_useful_end, default=0.0)
    for stage, first_layer, last_layer, layer_count in stage_spans:
        _record_pp_idle_row(
            result,
            event="BWD_PP_BUBBLE",
            start=stage_last_useful_end[stage],
            end=useful_makespan,
            stage=stage,
            microbatch=0,
            first_layer=first_layer,
            last_layer=last_layer,
            layer_count=layer_count,
            lane=stage_idle_lane(stage),
            topology_candidate_tag=(
                f"stage={stage}|microbatch=0|phase=bwd|reason=drain_bubble_after_last_useful_work"
            ),
        )

    result.makespan = useful_makespan
    return result


def build_time_series_overlap_rows(inputs, perf_model, deepflow_outputs) -> List[dict]:
    params = _parameters(inputs)
    schedule = build_overlap_schedule(inputs, perf_model)

    layers = int(params["layers"])
    parallel_divisor = max(
        1.0,
        float(
            params["data_parallel_degree"]
            * params["tensor_parallel_degree"]
            * params["pipeline_parallel_degree"]
        ),
    )

    fwd_compute = float(perf_model.compute_time_forward_pass()) / parallel_divisor
    weight_update = float(perf_model.weight_update_time()) / parallel_divisor

    fwd_tp_intra = float(perf_model.forward_tensor_model_intra())
    fwd_tp_inter = float(perf_model.forward_tensor_parallel_inter())
    bwd_tp_intra = float(perf_model.backward_tensor_model_intra())
    bwd_tp_inter = float(perf_model.backward_tensor_parallel_inter())

    fwd_pp_intra = float(perf_model.forward_pipeline_parallel_intra())
    fwd_pp_inter = float(perf_model.forward_pipeline_parallel_inter())
    bwd_pp_intra = float(perf_model.backward_pipeline_parallel_intra())
    bwd_pp_inter = float(perf_model.backward_pipeline_parallel_inter())

    activation_volume = (
        float(params["microbatch_size"])
        * float(params["hidden_layer_dimension_for_attention_sublayers"])
        * float(params["attention_heads"])
        * float(params["context"])
        * float(params["activation_precision"])
        / 8.0
    )
    error_volume = (
        float(params["microbatch_size"])
        * float(params["context"])
        * float(params["hidden_layer_dimension_for_attention_sublayers"])
        * float(params["attention_heads"])
        * float(params["gradient_precision"])
        / 8.0
    )

    fwd_tp_intra_bytes = activation_volume / max(1, int(params["inter_node_tensor_parallel_degree"]))
    fwd_tp_inter_bytes = activation_volume / max(1, int(params["intra_node_tensor_parallel_degree"]))
    bwd_tp_intra_bytes = error_volume / max(1, int(params["inter_node_tensor_parallel_degree"]))
    bwd_tp_inter_bytes = error_volume / max(1, int(params["intra_node_tensor_parallel_degree"]))

    rows: List[dict] = []

    main_cursor = 0.0
    forward_comm_cursor = 0.0
    for layer_idx in range(layers):
        layer_start = max(main_cursor, forward_comm_cursor)
        layer_end = layer_start + fwd_compute
        _add_csv_event(
            rows,
            event="FWD_LAYER_COMPUTE",
            event_type="Compute",
            start=layer_start,
            end=layer_end,
            lane="MAIN_SERIAL",
            layer=layer_idx,
            scheduler_explicit="no",
            ordering_source="phase_scaffold_serial",
            topology_candidate_tag="NONE",
        )
        main_cursor = layer_end

        comm_cursor = layer_end
        if fwd_tp_intra > 0.0:
            comm_end = comm_cursor + fwd_tp_intra
            _add_csv_event(
                rows,
                event="FWD_TP_COMM_INTRA",
                event_type="Comm",
                start=comm_cursor,
                end=comm_end,
                bytes_to_transfer=fwd_tp_intra_bytes,
                collective_type="ALLREDUCE",
                parallelism="TP",
                locality="INTRA",
                degree=int(params["intra_node_tensor_parallel_degree"]),
                lane="MAIN_SERIAL",
                layer=layer_idx,
                scheduler_explicit="no",
                ordering_source="phase_scaffold_serial",
            )
            comm_cursor = comm_end
        if fwd_tp_inter > 0.0:
            comm_end = comm_cursor + fwd_tp_inter
            _add_csv_event(
                rows,
                event="FWD_TP_COMM_INTER",
                event_type="Comm",
                start=comm_cursor,
                end=comm_end,
                bytes_to_transfer=fwd_tp_inter_bytes,
                collective_type="ALLREDUCE",
                parallelism="TP",
                locality="INTER",
                degree=int(params["inter_node_tensor_parallel_degree"]),
                lane="MAIN_SERIAL",
                layer=layer_idx,
                scheduler_explicit="no",
                ordering_source="phase_scaffold_serial",
            )
            comm_cursor = comm_end
        forward_comm_cursor = comm_cursor

    forward_local_duration = max(main_cursor, forward_comm_cursor)
    forward_pp_boundary_duration = max(fwd_pp_intra, fwd_pp_inter)
    backward_base = forward_local_duration + forward_pp_boundary_duration
    shifted_compute_windows: List[ComputeWindow] = []
    for window in schedule.compute_windows:
        shifted_compute_windows.append(
            ComputeWindow(
                stage=window.stage,
                microbatch=window.microbatch,
                layer=window.layer,
                sublayer=window.sublayer,
                start=backward_base + float(window.start or 0.0),
                end=backward_base + float(window.end or 0.0),
            )
        )

    wait_windows = build_backward_wait_windows(schedule.compute_windows)
    shifted_wait_windows: List[WaitWindow] = []
    for window in wait_windows:
        shifted_wait_windows.append(
            WaitWindow(
                layer=window.layer,
                blocked_by_layer=window.blocked_by_layer,
                start=backward_base + float(window.start),
                end=backward_base + float(window.end),
            )
        )

    for window in shifted_compute_windows:
        _add_csv_event(
            rows,
            event=window.sublayer,
            event_type="Compute",
            start=float(window.start or 0.0),
            end=float(window.end or 0.0),
            lane="BWD_COMPUTE_EXPLICIT",
            layer=window.layer,
            scheduler_explicit="yes",
            ordering_source="phase3_backward_compute_split",
            topology_candidate_tag="NONE",
        )

    for window in shifted_wait_windows:
        _add_csv_event(
            rows,
            event="BWD_WAIT_FOR_TP",
            event_type="Wait",
            start=float(window.start),
            end=float(window.end),
            lane="BWD_WAIT_EXPLICIT",
            layer=window.layer,
            scheduler_explicit="yes",
            ordering_source="phase4b_tp_feedback_blocking",
            topology_candidate_tag=f"blocked_by_layer={window.blocked_by_layer}",
            topology_signature="none",
        )

    shifted_tp_lane_end = backward_base
    for window in schedule.tp_comm_windows:
        shifted_start = backward_base + float(window.start or 0.0)
        shifted_end = backward_base + float(window.end or 0.0)
        shifted_ready = backward_base + float(window.ready_time)
        shifted_request = backward_base + float(window.request_time or window.ready_time)
        bytes_to_transfer = bwd_tp_intra_bytes if window.locality == "INTRA" else bwd_tp_inter_bytes
        degree = (
            int(params["intra_node_tensor_parallel_degree"])
            if window.locality == "INTRA"
            else int(params["inter_node_tensor_parallel_degree"])
        )
        lane = "BWD_TP_INTRA_EXPLICIT" if window.locality == "INTRA" else "BWD_TP_INTER_EXPLICIT"
        if window.locality == "INTER" and shifted_start > shifted_request:
            _add_csv_event(
                rows,
                event="BWD_TP_WAIT_FOR_FABRIC",
                event_type="Wait",
                start=shifted_request,
                end=shifted_start,
                bytes_to_transfer="NONE",
                collective_type="NONE",
                parallelism="TP",
                locality="INTER",
                degree=degree,
                lane="SHARED_INTERCONNECT_WAIT",
                layer=window.layer,
                ready_time=shifted_request,
                scheduler_explicit="yes",
                ordering_source="phase4a_shared_fabric",
            )
        _add_csv_event(
            rows,
            event=f"BWD_TP_COMM_{window.locality}",
            event_type="Comm",
            start=shifted_start,
            end=shifted_end,
            bytes_to_transfer=bytes_to_transfer,
            collective_type="ALLREDUCE",
            parallelism="TP",
            locality=window.locality,
            degree=degree,
            lane=lane,
            layer=window.layer,
            ready_time=shifted_ready,
            raw_comm_duration=window.raw_comm_duration,
            hidden_comm_duration=window.hidden_comm_duration,
            exposed_comm_duration=window.exposed_comm_duration,
            scheduler_explicit="yes",
            ordering_source="phase3_bwd_tp_scheduler",
        )
        shifted_tp_lane_end = max(shifted_tp_lane_end, shifted_end)

    backward_compute_end = max((float(w.end or 0.0) for w in shifted_compute_windows), default=backward_base)
    backward_phase_end = max(backward_compute_end, shifted_tp_lane_end)
    backward_local_duration = max(0.0, backward_phase_end - backward_base)
    backward_pp_boundary_duration = max(bwd_pp_intra, bwd_pp_inter)

    pp_schedule_result = build_pp_schedule_result(inputs, perf_model, schedule=schedule)
    rows.extend(pp_schedule_result.rows)

    dp_intra_degree = int(params["intra_node_data_parallel_degree"])
    dp_inter_degree = int(params["inter_node_data_parallel_degree"])
    dp_total_degree = int(params["data_parallel_degree"])
    dp_lane_end = backward_base
    for window in schedule.dp_comm_windows:
        locality = "INTRA+INTER" if dp_intra_degree > 1 and dp_inter_degree > 1 else ("INTER" if dp_inter_degree > 1 else "INTRA")
        topology_tag = (
            f"parallelism=DP|unit={window.unit_kind}|locality={locality}|"
            f"degree_total={dp_total_degree}|degree_intra={dp_intra_degree}|degree_inter={dp_inter_degree}|"
            f"bytes={window.bytes}"
        )
        shifted_ready = backward_base + float(window.ready_time)
        shifted_request = backward_base + float(window.request_time or window.comm_start or 0.0)
        shifted_reconf_start = (
            backward_base + float(window.reconf_start)
            if window.reconf_start is not None
            else None
        )
        shifted_reconf_end = (
            backward_base + float(window.reconf_end)
            if window.reconf_end is not None
            else None
        )
        shifted_comm_start = backward_base + float(window.comm_start or 0.0)
        shifted_comm_end = backward_base + float(window.comm_end or 0.0)

        if shifted_reconf_start is not None and shifted_reconf_end is not None:
            _add_csv_event(
                rows,
                event="BWD_DP_RECONFIG",
                event_type="Reconfiguration",
                start=shifted_reconf_start,
                end=shifted_reconf_end,
                bytes_to_transfer="NONE",
                collective_type="NONE",
                parallelism="DP",
                locality=locality,
                degree=dp_total_degree,
                lane="DP_SCHED_EXPLICIT",
                layer="NONE",
                first_layer=window.first_layer if window.first_layer is not None else "NONE",
                last_layer=window.last_layer if window.last_layer is not None else "NONE",
                layer_count=window.layer_count,
                bucket_id=window.bucket_id,
                unit_kind=window.unit_kind,
                ready_time=shifted_ready,
                raw_comm_duration=window.raw_comm_duration,
                raw_reconf_duration=window.raw_reconf_duration,
                hidden_comm_duration=window.hidden_comm_duration,
                exposed_comm_duration=window.exposed_comm_duration,
                hidden_reconf_duration=window.hidden_reconf_duration,
                exposed_reconf_duration=window.exposed_reconf_duration,
                scheduler_explicit="yes",
                ordering_source="phase2_dp_scheduler",
                topology_candidate_tag=topology_tag,
            )

        if shifted_comm_start > shifted_request:
            _add_csv_event(
                rows,
                event="BWD_DP_WAIT_FOR_FABRIC",
                event_type="Wait",
                start=shifted_request,
                end=shifted_comm_start,
                bytes_to_transfer="NONE",
                collective_type="NONE",
                parallelism="DP",
                locality=locality,
                degree=dp_total_degree,
                lane="SHARED_INTERCONNECT_WAIT",
                layer="NONE",
                first_layer=window.first_layer if window.first_layer is not None else "NONE",
                last_layer=window.last_layer if window.last_layer is not None else "NONE",
                layer_count=window.layer_count,
                bucket_id=window.bucket_id,
                unit_kind=window.unit_kind,
                ready_time=shifted_request,
                scheduler_explicit="yes",
                ordering_source="phase4a_shared_fabric",
                topology_candidate_tag=topology_tag,
            )

        _add_csv_event(
            rows,
            event="BWD_DP_COMM",
            event_type="Comm",
            start=shifted_comm_start,
            end=shifted_comm_end,
            bytes_to_transfer=window.bytes,
            collective_type="ALLREDUCE",
            parallelism="DP",
            locality=locality,
            degree=dp_total_degree,
            lane="DP_SCHED_EXPLICIT",
            layer="NONE",
            first_layer=window.first_layer if window.first_layer is not None else "NONE",
            last_layer=window.last_layer if window.last_layer is not None else "NONE",
            layer_count=window.layer_count,
            bucket_id=window.bucket_id,
            unit_kind=window.unit_kind,
            ready_time=shifted_ready,
            raw_comm_duration=window.raw_comm_duration,
            raw_reconf_duration=window.raw_reconf_duration,
            hidden_comm_duration=window.hidden_comm_duration,
            exposed_comm_duration=window.exposed_comm_duration,
            hidden_reconf_duration=window.hidden_reconf_duration,
            exposed_reconf_duration=window.exposed_reconf_duration,
            scheduler_explicit="yes",
            ordering_source="phase2_dp_scheduler",
            topology_candidate_tag=topology_tag,
        )
        dp_lane_end = max(dp_lane_end, shifted_comm_end)

    weight_update_base = max(backward_phase_end + backward_pp_boundary_duration, dp_lane_end)
    weight_cursor = weight_update_base
    for layer_idx in range(layers):
        event_end = weight_cursor + weight_update
        _add_csv_event(
            rows,
            event="WEIGHT_UPDATE",
            event_type="Compute",
            start=weight_cursor,
            end=event_end,
            lane="MAIN_SERIAL",
            layer=layer_idx,
            scheduler_explicit="no",
            ordering_source="phase_scaffold_serial",
            topology_candidate_tag="NONE",
        )
        weight_cursor = event_end

    rows.sort(key=lambda row: (float(row["start time"]), float(row["end time"]), row["Lane"], row["Event"]))
    _annotate_previous_event_fields(rows)
    return rows


def build_time_series_overlap_csv(inputs, perf_model, deepflow_outputs) -> str:
    rows = build_time_series_overlap_rows(inputs, perf_model, deepflow_outputs)
    params = _parameters(inputs)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Total batches",
            params["number_of_batches"],
            "Explicit schedule unit",
            "one optimization step",
            "DP overlap mode",
            perf_model.dp_overlap_mode_name() if hasattr(perf_model, "dp_overlap_mode_name") else "UNKNOWN",
        ]
    )
    writer.writerow(
        [
            "Event",
            "Type",
            "start time",
            "end time",
            "duration",
            "Bytes to be transferred",
            "Collective type",
            "Parallelism",
            "Locality",
            "Degree",
            "Lane",
            "Step",
            "Layer index",
            "First layer",
            "Last layer",
            "Layer count",
            "Bucket id",
            "Unit kind",
            "Ready time",
            "Raw comm duration",
            "Raw reconf duration",
            "Hidden comm duration",
            "Exposed comm duration",
            "Hidden reconf duration",
            "Exposed reconf duration",
            "Stage index",
            "Microbatch index",
            "Scheduler explicit",
            "Event provenance",
            "Ordering source",
            "topology_signature",
            "Topology candidate tag",
            "previous_event_type",
            "previous_topology_signature",
            "gap_from_previous_s",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row["Event"],
                row["Type"],
                row["start time"],
                row["end time"],
                row["duration"],
                row["Bytes to be transferred"],
                row["Collective type"],
                row["Parallelism"],
                row["Locality"],
                row["Degree"],
                row["Lane"],
                row["Step"],
                row["Layer index"],
                row["First layer"],
                row["Last layer"],
                row["Layer count"],
                row["Bucket id"],
                row["Unit kind"],
                row["Ready time"],
                row["Raw comm duration"],
                row["Raw reconf duration"],
                row["Hidden comm duration"],
                row["Exposed comm duration"],
                row["Hidden reconf duration"],
                row["Exposed reconf duration"],
                row["Stage index"],
                row["Microbatch index"],
                row["Scheduler explicit"],
                row["Event provenance"],
                row["Ordering source"],
                row["topology_signature"],
                row["Topology candidate tag"],
                row["previous_event_type"],
                row["previous_topology_signature"],
                row["gap_from_previous_s"],
            ]
        )
    return output.getvalue()
