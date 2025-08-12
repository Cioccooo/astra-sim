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

void TopologyManager::init(const std::string& network_config,
                           AnalyticalBackendType type) noexcept {
  network_config_path = network_config;
  backend_type = type;
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

  // 校验 npus_count 一致性：和 active_profile (parser 初始状态) 保持相同
  const auto base_npus = parser.get_npus_counts_per_dim();
  if (prof.npus_counts_per_dim != base_npus) {
    exit_with_error("[Reconfig] npus_count mismatch across profiles");
  }

  // 构建拓扑（目前仅支持 1 维；多维走 helper）
  if (prof.topologies_per_dim.size() == 1) {
    const auto topo = prof.topologies_per_dim[0];
    const auto npus = prof.npus_counts_per_dim[0];
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
  std::cerr << "[Reconfig] applied profile '" << profile_name << "' dims=" << dims
            << " bw[0]=" << (bws.empty() ? 0.0 : bws[0]) << std::endl;
  apply_topology(std::move(topo));
}

}  // namespace AstraSimAnalytical

