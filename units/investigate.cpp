#include <pybind11/pybind11.h>
#include <sstream>
#include "units.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

std::string getBaseSIUnits(std::string un) {
    auto pu = UNITS_NAMESPACE::unit_from_string(un);
    auto u = pu.base_units();
    int meter = u.meter();
    int kg = u.kg();
    int second = u.second();
    int ampere = u.ampere();
    int kelvin = u.kelvin();
    int mole   = u.mole();
    int candela = u.candela();
    double mult = pu.multiplier();
    std::ostringstream ss;
    ss << "{\"m\":" <<meter<<",";
    ss << "\"kg\":" <<kg<<",";
    ss << "\"s\":" <<second<<",";
    ss << "\"A\":" <<ampere<<",";
    ss << "\"K\":" <<kelvin<<",";
    ss << "\"mol\":" <<mole<<",";
    ss << "\"cd\":" <<candela<<",";
    ss << "\"mult\":" <<mult<<"}";
    std::string res(ss.str());
    return res;
}

namespace py = pybind11;

PYBIND11_MODULE(ftuutils_units, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: ftuutils

        .. autosummary::
           :toctree: _generate

           getBaseSIUnits
    )pbdoc";

    m.def("getBaseSIUnits", &getBaseSIUnits, R"pbdoc(
        Get the precise SI unit associated with the string to define 
        cellml units
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}