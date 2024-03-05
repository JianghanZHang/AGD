///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "grg/python.hpp"
#include "grg/grg.hpp"

namespace grg {

namespace bp = boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverGRG_solves, SolverGRG::solve, 0, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverGRG_computeDirections, SolverGRG::computeDirection, 0, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverGRG_trySteps, SolverGRG::tryStep, 0, 1)


void exposeSolverGRG() {
  bp::register_ptr_to_python<boost::shared_ptr<SolverGRG> >();

  bp::class_<SolverGRG, bp::bases<crocoddyl::SolverAbstract> >(
      "SolverGRG",
      "SQP solver.\n\n"
      "The GRG solver.\n"
      ":param shootingProblem: shooting problem (list of action models along trajectory.)",
      bp::init<boost::shared_ptr<crocoddyl::ShootingProblem> >(bp::args("self", "problem"),
                                                    "Initialize the vector dimension.\n\n"
                                                    ":param problem: shooting problem."))
      .def("solve", &SolverGRG::solve,
           SolverGRG_solves(
               bp::args("self", "init_xs", "init_us", "maxiter", "isFeasible"),
               "Compute the optimal trajectory xopt, uopt as lists of T+1 and T terms.\n\n"
               "From an initial guess init_xs,init_us (feasible or not), iterate\n"
               "over computeDirection and tryStep until stoppingCriteria is below\n"
               "threshold. It also describes the globalization strategy used\n"
               "during the numerical optimization.\n"
               ":param init_xs: initial guess for state trajectory with T+1 elements (default [])\n"
               ":param init_us: initial guess for control trajectory with T elements (default []).\n"
               ":param maxiter: maximum allowed number of iterations (default 100).\n"
               ":param isFeasible: true if the init_xs are obtained from integrating the init_us (rollout) (default "
               "False).\n"
               ":returns the optimal trajectory xopt, uopt and a boolean that describes if convergence was reached."))
      .def_readwrite("xs_try", &SolverGRG::xs_try_, "xs try")
      .def_readwrite("us_try", &SolverGRG::us_try_, "us try")
      .def_readwrite("cost_try", &SolverGRG::cost_try_, "cost try")
      .def_readwrite("fs_try", &SolverGRG::fs_try_, "fs_try")

      .add_property("KKT", bp::make_function(&SolverGRG::get_KKT),
                    "KKT residual norm")

      .add_property("gap_norm", bp::make_function(&SolverGRG::get_gap_norm),
                     "gap norm")

      .add_property("x_grad_norm", bp::make_function(&SolverGRG::get_xgrad_norm), "x_grad_norm")

      .add_property("u_grad_norm", bp::make_function(&SolverGRG::get_ugrad_norm), "u_grad_norm")

      .add_property("const_step_length", bp::make_function(&SolverGRG::getConstStepLength), bp::make_function(&SolverGRG::setConstStepLength),
                    "Constant step length (default: 1.)")

      .add_property("use_line_search", bp::make_function(&SolverGRG::get_use_line_search), bp::make_function(&SolverGRG::set_use_line_search),
                    "Use the line search criteria (default: False)")
    
      .add_property("beta1", bp::make_function(&SolverGRG::get_beta1), bp::make_function(&SolverGRG::set_beta1), "Beta1 (default: 0.9)")

      .add_property("beta2", bp::make_function(&SolverGRG::get_beta2), bp::make_function(&SolverGRG::set_beta2), "Beta2 (default: 0.999)")

      .add_property("correct_bias", bp::make_function(&SolverGRG::get_correct_bias), bp::make_function(&SolverGRG::set_correct_bias),
                    "Correct the bias in the line search (default: False)")

      .add_property("m", bp::make_function(&SolverGRG::get_m), bp::make_function(&SolverGRG::set_m),
                    "momentum in Adaptive Momentum methods")

      .add_property("v", bp::make_function(&SolverGRG::get_v), bp::make_function(&SolverGRG::set_v),
                    "2nd-order momentum in Adaptive Momentum methods")

      .add_property("with_callbacks", bp::make_function(&SolverGRG::getCallbacks), bp::make_function(&SolverGRG::setCallbacks),
                    "Activates the callbacks when true (default: False)")
    
      .add_property("mu", bp::make_function(&SolverGRG::get_mu), bp::make_function(&SolverGRG::set_mu),
                    "Penalty term for dynamic violation in the merit function (default: 0.)")

      .add_property("termination_tolerance", bp::make_function(&SolverGRG::get_termination_tolerance), bp::make_function(&SolverGRG::set_termination_tolerance),
                    "Termination criteria to exit the iteration (default: 1e-6)");
     
}

}  // namespace mim_solvers