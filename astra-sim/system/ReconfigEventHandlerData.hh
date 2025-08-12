/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __RECONFIG_EVENT_HANDLER_DATA_HH__
#define __RECONFIG_EVENT_HANDLER_DATA_HH__

#include <string>

#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

class ReconfigEventHandlerData : public BasicEventHandlerData {
  public:
    ReconfigEventHandlerData(int sys_id,
                             EventType event,
                             const std::string& target_profile,
                             bool has_bw_override,
                             double bw_gbps,
                             bool has_lat_override,
                             double lat_ns)
        : BasicEventHandlerData(sys_id, event),
          target_profile(target_profile),
          has_bw_override(has_bw_override),
          bw_gbps(bw_gbps),
          has_lat_override(has_lat_override),
          lat_ns(lat_ns) {}

    std::string target_profile;
    bool has_bw_override;
    double bw_gbps;
    bool has_lat_override;
    double lat_ns;
};

}  // namespace AstraSim

#endif /* __RECONFIG_EVENT_HANDLER_DATA_HH__ */

