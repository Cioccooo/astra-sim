/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CommonNetworkApi.hh"
#include <astra-network-analytical/congestion_aware/Topology.h>
#include <string>

using namespace AstraSim;
using namespace AstraSimAnalytical;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace AstraSimAnalyticalCongestionAware {

class CongestionAwareNetworkApi final : public CommonNetworkApi {
  public:
    /**
     * Set the topology to be used.
     *
     * @param topology_ptr pointer to the topology
     */
    static void set_topology(std::shared_ptr<Topology> topology_ptr) noexcept;

    /**
     * Set routing options parsed from the network configuration.
     */
    static void set_routing_config(std::string routing,
                                   std::string ecmp_split,
                                   int ecmp_max_paths,
                                   bool ecmp_log) noexcept;

    /**
     * Constructor.
     *
     * @param rank id of the API
     */
    explicit CongestionAwareNetworkApi(int rank) noexcept;

    /**
     * Implement sim_send of AstraNetworkAPI.
     */
    int sim_send(void* buffer,
                 uint64_t count,
                 int type,
                 int dst,
                 int tag,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg) override;

  private:
    /// topology
    static std::shared_ptr<Topology> topology;

    static bool ecmp_enabled;

    static int ecmp_max_paths;

    static bool ecmp_log;

    static void process_ecmp_chunk_arrival(void* args) noexcept;
};

}  // namespace AstraSimAnalyticalCongestionAware
