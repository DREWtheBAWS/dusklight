#include "geometry_collector.hpp"
#include <aurora/geometry_capture.h>

namespace dusk::rtao {

void GeometryCollector::install() {
    aurora_set_geometry_capture(&GeometryCollector::on_capture, this);
}

} // namespace dusk::rtao
