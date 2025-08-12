/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <memory>
#include <optional>
#include <string>

#include <astra-network-analytical/common/Type.h>

namespace NetworkAnalytical {
class Topology;
}

namespace AstraSimAnalytical {

enum class AnalyticalBackendType { CongestionUnaware = 0, CongestionAware };

struct ReconfigOverrides {
  bool has_bandwidth_override = false;
  double bandwidth_GBps = 0.0;
  bool has_latency_override = false;
  double latency_ns = 0.0;
};

class TopologyManager {
 public:
  // Initialize with the original network configuration file and backend type
  static void init(const std::string& network_config_path,
                   AnalyticalBackendType backend_type) noexcept;

  // Switch to a profile by name, applying optional overrides
  static void switch_to_profile(const std::string& profile_name,
                                const ReconfigOverrides& overrides);

  // Inspect a profile (topology/bw/lat) without applying it
  struct ProfileDesc {
    std::string topology_name;
    double bandwidth_gbps;
    double latency_ns;
  };
  static ProfileDesc describe_profile(const std::string& profile_name);

 private:
  static std::string network_config_path;
  static AnalyticalBackendType backend_type;
  static int npus_count_current;

  // Build a topology shared_ptr from a profile in example_network.yml
  static std::shared_ptr<NetworkAnalytical::Topology>
  build_topology_from_profile(const std::string& profile_name,
                              const ReconfigOverrides& overrides);

  // Apply the newly created topology into the active Analytical backend
  static void apply_topology(
      std::shared_ptr<NetworkAnalytical::Topology> topology_ptr);
};

}  // namespace AstraSimAnalytical

