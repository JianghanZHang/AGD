///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "grg/python.hpp"

BOOST_PYTHON_MODULE(grg_pywrap) { 

    namespace bp = boost::python;

    bp::import("crocoddyl");

    grg::exposeSolverGRG();
}
