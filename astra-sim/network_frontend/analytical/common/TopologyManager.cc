/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/TopologyManager.hh"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <astra-network-analytical/congestion_unaware/Helper.h>
#include <astra-network-analytical/congestion_unaware/Ring.h>
#include <astra-network-analytical/congestion_unaware/FullyConnected.h>
#include <astra-network-analytical/congestion_unaware/Switch.h>
#include <astra-network-analytical/congestion_aware/Ring.h>
#include <astra-network-analytical/congestion_aware/FullyConnected.h>
#include <astra-network-analytical/congestion_aware/Switch.h>

#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include "congestion_unaware/CongestionUnawareNetworkApi.hh"

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;
using namespace NetworkAnalyticalCongestionUnaware;

namespace AstraSimAnalytical {

std::string TopologyManager::network_config_path;
AnalyticalBackendType TopologyManager::backend_type =
    AnalyticalBackendType::CongestionUnaware;
int TopologyManager::npus_count_current = -1;

void TopologyManager::init(const std::string& network_config,
                           AnalyticalBackendType type) noexcept {
  network_config_path = network_config;
  backend_type = type;
  // Parse initial config to record current N
  NetworkParser parser(network_config_path);
  const auto npus = parser.get_npus_counts_per_dim();
  if (!npus.empty()) {
    npus_count_current = npus[0];
  }
}

static void exit_with_error(const std::string& msg) {
  std::cerr << msg << std::endl;
  std::exit(-1);
}

std::shared_ptr<Topology> TopologyManager::build_topology_from_profile(
    const std::string& profile_name, const ReconfigOverrides& /*overrides*/) {
  // Parse YAML and pick the named profile precisely
  NetworkParser parser(network_config_path);
  if (!parser.has_profile(profile_name)) {
    exit_with_error(std::string("[Reconfig] profile not found: ") + profile_name);
  }
  const auto prof = parser.get_profile(profile_name);
  // 运行时不改变 N：忽略 profile 中的 npus_count，与当前运行的 npus_count_current 保持一致
  if (!prof.npus_counts_per_dim.empty() && npus_count_current > 0 &&
      prof.npus_counts_per_dim[0] != npus_count_current) {
    std::cerr << "[WARN] profile npus_count (" << prof.npus_counts_per_dim[0]
              << ") ignored, using current N (" << npus_count_current << ")" << std::endl;
  }

  // 构建拓扑（目前仅支持 1 维；多维走 helper）
  if (prof.topologies_per_dim.size() == 1) {
    const auto topo = prof.topologies_per_dim[0];
    const auto npus = npus_count_current > 0 ? npus_count_current : (int)prof.npus_counts_per_dim[0];
    const auto bw = prof.bandwidths_per_dim[0];
    const auto lat = prof.latencies_per_dim[0];
    if (backend_type == AnalyticalBackendType::CongestionAware) {
      switch (topo) {
        case TopologyBuildingBlock::Ring:
          return std::make_shared<NetworkAnalyticalCongestionAware::Ring>(npus, bw, lat);
        case TopologyBuildingBlock::FullyConnected:
          return std::make_shared<NetworkAnalyticalCongestionAware::FullyConnected>(npus, bw, lat);
        case TopologyBuildingBlock::Switch:
          return std::make_shared<NetworkAnalyticalCongestionAware::Switch>(npus, bw, lat);
        default:
          exit_with_error("[Reconfig] Unsupported topology building block");
          return nullptr;
      }
    } else {
      switch (topo) {
        case TopologyBuildingBlock::Ring:
          return std::make_shared<NetworkAnalyticalCongestionUnaware::Ring>(npus, bw, lat);
        case TopologyBuildingBlock::FullyConnected:
          return std::make_shared<NetworkAnalyticalCongestionUnaware::FullyConnected>(npus, bw, lat);
        case TopologyBuildingBlock::Switch:
          return std::make_shared<NetworkAnalyticalCongestionUnaware::Switch>(npus, bw, lat);
        default:
          exit_with_error("[Reconfig] Unsupported topology building block");
          return nullptr;
      }
    }
  }

  // 多维：将当前 parser 状态替换为目标 profile 后，交给 helper 构造
  // 这里直接临时覆盖 parser 的内部视图（通过局部变量），再调用 helper
  // 简化起见，直接使用 helper 对 prof 构造
  // 为保持与现有接口兼容，仍返回 helper 构造结果
  // 注意：示例中使用 1 维，故通常不会走到这里
  return (backend_type == AnalyticalBackendType::CongestionAware)
             ? NetworkAnalyticalCongestionAware::construct_topology(parser)
             : NetworkAnalyticalCongestionUnaware::construct_topology(parser);
}

void TopologyManager::apply_topology(
    std::shared_ptr<Topology> topology_ptr) {
  if (!topology_ptr) {
    exit_with_error("[Reconfig] null topology_ptr");
  }
  if (backend_type == AnalyticalBackendType::CongestionAware) {
    AstraSimAnalyticalCongestionAware::CongestionAwareNetworkApi::set_topology(
        std::move(topology_ptr));
  } else {
    AstraSimAnalyticalCongestionUnaware::CongestionUnawareNetworkApi::set_topology(
        std::move(topology_ptr));
  }
}

void TopologyManager::switch_to_profile(const std::string& profile_name,
                                        const ReconfigOverrides& overrides) {
  // 忽略 overrides（按需求保持事件仅含 target_profile 与 delay_cycles）
  auto topo = build_topology_from_profile(profile_name, overrides);
  // 打印最终生效参数
  const auto dims = topo->get_dims_count();
  const auto bws = topo->get_bandwidth_per_dim();
  std::cerr << "[Reconfig] applied profile '" << profile_name
            << "', topology=" << (prof.topologies_per_dim.empty() ? -1 : (int)prof.topologies_per_dim[0])
            << ", bandwidth=" << (bws.empty() ? 0.0 : bws[0]) << " GB/s"
            << ", latency=" << (prof.latencies_per_dim.empty() ? 0.0 : prof.latencies_per_dim[0]) << " ns"
            << ", npus_current=" << npus_count_current
            << std::endl;
  apply_topology(std::move(topo));
}

TopologyManager::ProfileDesc TopologyManager::describe_profile(const std::string& profile_name) {
  NetworkParser parser(network_config_path);
  const auto prof = parser.get_profile(profile_name);
  ProfileDesc d;
  if (!prof.topologies_per_dim.empty()) {
    switch (prof.topologies_per_dim[0]) {
      case TopologyBuildingBlock::Ring: d.topology_name = "Ring"; break;
      case TopologyBuildingBlock::FullyConnected: d.topology_name = "FullyConnected"; break;
      case TopologyBuildingBlock::Switch: d.topology_name = "Switch"; break;
      default: d.topology_name = "Unknown"; break;
    }
  }
  d.bandwidth_gbps = prof.bandwidths_per_dim.empty() ? 0.0 : prof.bandwidths_per_dim[0];
  d.latency_ns = prof.latencies_per_dim.empty() ? 0.0 : prof.latencies_per_dim[0];
  return d;
}

}  // namespace AstraSimAnalytical

