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
    const std::string& profile_name, const ReconfigOverrides& overrides) {
  // For MVP we reuse the same YAML file; profiles are pre-resolved by user.
  // Here we simply parse the YAML via NetworkParser, which expects the
  // top-level keys (topology/npus_count/bandwidth/latency). The profile
  // selection and overriding is assumed to be done by providing a temporary
  // YAML path equal to network_config_path (already set to desired profile).
  // If overrides are provided, we will adjust the parsed values before build.
  NetworkParser parser(network_config_path);

  // If overrides exist and dims_count==1 (Analytical examples), apply them by
  // creating a shallow clone of parsed values and constructing topology
  // directly. Otherwise, defer to construct_topology(parser).
  const auto dims = parser.get_dims_count();
  const auto tops = parser.get_topologies_per_dim();
  const auto npus = parser.get_npus_counts_per_dim();
  auto bws = parser.get_bandwidths_per_dim();
  auto lats = parser.get_latencies_per_dim();

  if (dims == 1) {
    if (overrides.has_bandwidth_override) {
      bws[0] = overrides.bandwidth_GBps;
    }
    if (overrides.has_latency_override) {
      lats[0] = overrides.latency_ns;
    }
    if (backend_type == AnalyticalBackendType::CongestionAware) {
      switch (tops[0]) {
        case TopologyBuildingBlock::Ring:
          return std::make_shared<NetworkAnalyticalCongestionAware::Ring>(
              npus[0], bws[0], lats[0]);
        case TopologyBuildingBlock::FullyConnected:
          return std::make_shared<NetworkAnalyticalCongestionAware::FullyConnected>(
              npus[0], bws[0], lats[0]);
        case TopologyBuildingBlock::Switch:
          return std::make_shared<NetworkAnalyticalCongestionAware::Switch>(
              npus[0], bws[0], lats[0]);
        default:
          exit_with_error("[Reconfig] Unsupported topology building block");
          return nullptr;
      }
    } else {
      switch (tops[0]) {
        case TopologyBuildingBlock::Ring:
          return std::make_shared<NetworkAnalyticalCongestionUnaware::Ring>(
              npus[0], bws[0], lats[0]);
        case TopologyBuildingBlock::FullyConnected:
          return std::make_shared<NetworkAnalyticalCongestionUnaware::FullyConnected>(
              npus[0], bws[0], lats[0]);
        case TopologyBuildingBlock::Switch:
          return std::make_shared<NetworkAnalyticalCongestionUnaware::Switch>(
              npus[0], bws[0], lats[0]);
        default:
          exit_with_error("[Reconfig] Unsupported topology building block");
          return nullptr;
      }
    }
  }

  // Multi-dim path: delegate to helpers, which will use parser's values.
  if (backend_type == AnalyticalBackendType::CongestionAware) {
    return NetworkAnalyticalCongestionAware::construct_topology(parser);
  }
  return NetworkAnalyticalCongestionUnaware::construct_topology(parser);
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
  auto topo = build_topology_from_profile(profile_name, overrides);
  apply_topology(std::move(topo));
}

}  // namespace AstraSimAnalytical

