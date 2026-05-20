/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/congestion_aware/Chunk.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <tuple>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> CongestionAwareNetworkApi::topology;
bool CongestionAwareNetworkApi::ecmp_enabled = false;
int CongestionAwareNetworkApi::ecmp_max_paths = 0;
bool CongestionAwareNetworkApi::ecmp_log = false;

namespace {
struct EcmpArrivalState {
    int tag;
    int src;
    int dst;
    uint64_t count;
    int chunk_id;
    int remaining_subchunks;
};

std::string route_to_string(const Route& route) {
    auto stream = std::ostringstream();
    stream << "[";
    auto first = true;
    for (const auto& device : route) {
        if (!first) {
            stream << ",";
        }
        first = false;
        stream << device->get_id();
    }
    stream << "]";
    return stream.str();
}
}  // namespace

void CongestionAwareNetworkApi::set_topology(
    std::shared_ptr<Topology> topology_ptr) noexcept {
    assert(topology_ptr != nullptr);

    // move topology
    CongestionAwareNetworkApi::topology = std::move(topology_ptr);

    // set topology-related values
    CongestionAwareNetworkApi::dims_count =
        CongestionAwareNetworkApi::topology->get_dims_count();
    CongestionAwareNetworkApi::bandwidth_per_dim =
        CongestionAwareNetworkApi::topology->get_bandwidth_per_dim();
}

void CongestionAwareNetworkApi::set_routing_config(
    const std::string routing,
    const std::string ecmp_split,
    const int configured_ecmp_max_paths,
    const bool configured_ecmp_log) noexcept {
    assert(routing == "deterministic" || routing == "ecmp");
    assert(ecmp_split == "equal_bytes");
    assert(configured_ecmp_max_paths >= 0);

    CongestionAwareNetworkApi::ecmp_enabled = (routing == "ecmp");
    CongestionAwareNetworkApi::ecmp_max_paths = configured_ecmp_max_paths;
    CongestionAwareNetworkApi::ecmp_log = configured_ecmp_log;
}

CongestionAwareNetworkApi::CongestionAwareNetworkApi(const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
}

int CongestionAwareNetworkApi::sim_send(void* const buffer,
                                        const uint64_t count,
                                        const int type,
                                        const int dst,
                                        const int tag,
                                        sim_request* const request,
                                        void (*msg_handler)(void*),
                                        void* const fun_arg) {
    // query chunk id
    const auto src = sim_comm_get_rank();
    const auto chunk_id =
        CongestionAwareNetworkApi::chunk_id_generator.create_send_chunk_id(
            tag, src, dst, count);

    // search tracker
    const auto entry =
        callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // recv operation already issued.
        // register send callback
        entry.value()->register_send_callback(msg_handler, fun_arg);
    } else {
        // recv operation not issued yet
        // create new entry and insert callback
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_send_callback(msg_handler, fun_arg);
    }

    auto routes = std::vector<Route>();
    if (CongestionAwareNetworkApi::ecmp_enabled) {
        routes = topology->routes(src, dst);
        if (CongestionAwareNetworkApi::ecmp_max_paths > 0 &&
            static_cast<int>(routes.size()) > CongestionAwareNetworkApi::ecmp_max_paths) {
            routes.resize(CongestionAwareNetworkApi::ecmp_max_paths);
        }
        // A zero-byte subchunk is invalid, so use at most one path per byte.
        if (routes.size() > count) {
            routes.resize(count);
        }
    }
    if (routes.empty()) {
        routes.push_back(topology->route(src, dst));
    }

    if (routes.size() == 1) {
        // create chunk
        auto chunk_arrival_arg = std::tuple(tag, src, dst, count, chunk_id);
        auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
        const auto arg_ptr = static_cast<void*>(arg.release());
        auto chunk = std::make_unique<Chunk>(
            count, routes.front(), CongestionAwareNetworkApi::process_chunk_arrival,
            arg_ptr);

        // initiate transmission from src -> dst.
        topology->send(std::move(chunk));
    } else {
        const auto paths_count = routes.size();
        const auto base_size = count / paths_count;
        const auto remainder = count % paths_count;
        auto* state = new EcmpArrivalState{
            tag, src, dst, count, chunk_id, static_cast<int>(paths_count)};
        if (CongestionAwareNetworkApi::ecmp_log) {
            std::cout << "[ecmp] tag=" << tag << " src=" << src << " dst=" << dst
                      << " bytes=" << count << " paths=" << paths_count;
        }
        for (auto i = 0ul; i < paths_count; i++) {
            const auto subchunk_size = base_size + (i < remainder ? 1 : 0);
            assert(subchunk_size > 0);
            if (CongestionAwareNetworkApi::ecmp_log) {
                std::cout << " subchunk" << i << "=" << subchunk_size
                          << " route=" << route_to_string(routes[i]);
            }
            auto chunk = std::make_unique<Chunk>(
                subchunk_size, routes[i],
                CongestionAwareNetworkApi::process_ecmp_chunk_arrival,
                static_cast<void*>(state));
            topology->send(std::move(chunk));
        }
        if (CongestionAwareNetworkApi::ecmp_log) {
            std::cout << std::endl;
        }
    }

    // return
    return 0;
}

void CongestionAwareNetworkApi::process_ecmp_chunk_arrival(void* args) noexcept {
    assert(args != nullptr);

    auto* const state = static_cast<EcmpArrivalState*>(args);
    assert(state->remaining_subchunks > 0);
    state->remaining_subchunks--;
    if (state->remaining_subchunks > 0) {
        return;
    }

    auto& tracker = CommonNetworkApi::get_callback_tracker();
    const auto entry = tracker.search_entry(
        state->tag, state->src, state->dst, state->count, state->chunk_id);
    assert(entry.has_value());

    if (entry.value()->both_callbacks_registered()) {
        entry.value()->invoke_send_handler();
        entry.value()->invoke_recv_handler();
        tracker.pop_entry(state->tag, state->src, state->dst, state->count, state->chunk_id);
    } else {
        entry.value()->invoke_send_handler();
        entry.value()->set_transmission_finished();
    }
    delete state;
}
