#include <boost/python.hpp>

#include "grg/grg.hpp"
#include "grg/grg_adahessian.hpp"
#include "grg/grg_nonlinear.hpp"

namespace grg{
    void exposeSolverGRG();
    void exposeSolverGRG_ADAHESSIAN();
    void exposeSolverGRG_nonlinear();
} // namespace mim_solvers
