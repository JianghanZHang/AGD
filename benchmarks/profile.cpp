#include <iostream>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>

#include "grg/grg.hpp"
#include "grg/grg_adahessian.hpp"

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/multibody/fwd.hpp"

#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"

#include "crocoddyl/multibody/residuals/control-gravity.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"

#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"


#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"

#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

#include "crocoddyl/core/utils/stop-watch.hpp"

#include "timings.hpp"
// #include <core/utils/stop-watch.hpp>

#define  EXAMPLE_ROBOT_DATA_MODEL_DIR "/opt/openrobots/share/example-robot-data/robots"



int main(){

    // LOADING THE ROBOT AND INIT VARIABLES

    auto urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/ur_description/urdf/ur5_robot.urdf";

    boost::shared_ptr<pinocchio::Model> rmodel = boost::make_shared<pinocchio::Model>();
    pinocchio::urdf::buildModel(urdf_path, *rmodel.get());
    const int nq = rmodel->nq;
    const int nv = rmodel->nv;
    const int nu = nv;
    Eigen::VectorXd x0; x0.resize(nq + nu); x0.setZero();

    // STATE AND ACTUATION VARIABLES

    boost::shared_ptr<crocoddyl::StateMultibody> state = boost::make_shared<crocoddyl::StateMultibody>(rmodel);
    boost::shared_ptr<crocoddyl::ActuationModelFull> actuation = boost::make_shared<crocoddyl::ActuationModelFull>(state);
     
    boost::shared_ptr<crocoddyl::ResidualModelControlGrav> uResidual = boost::make_shared<crocoddyl::ResidualModelControlGrav>(state); 
    boost::shared_ptr<crocoddyl::CostModelResidual> uRegCost = boost::make_shared<crocoddyl::CostModelResidual>(state, uResidual); 
    
    boost::shared_ptr<crocoddyl::ResidualModelState> xResidual = boost::make_shared<crocoddyl::ResidualModelState>(state, x0);
    boost::shared_ptr<crocoddyl::CostModelResidual> xRegCost = boost::make_shared<crocoddyl::CostModelResidual>(state, xResidual); 
     
    // END EFFECTOR FRAME TRANSLATION COST

    const int endeff_frame_id = rmodel->getFrameId("tool0");
    Eigen::Vector3d endeff_translation = {0.4, 0.4, 0.4};
    boost::shared_ptr<crocoddyl::ResidualModelFrameTranslation> frameTranslationResidual = boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(
                                                                                                    state, 
                                                                                                    endeff_frame_id, 
                                                                                                    endeff_translation    
                                                                                                );
    boost::shared_ptr<crocoddyl::CostModelResidual> frameTranslationCost = boost::make_shared<crocoddyl::CostModelResidual>(state, frameTranslationResidual); 

    // DEFINE CONSTRAINTS
    boost::shared_ptr<crocoddyl::ResidualModelFrameTranslation> frameTranslationConstraintResidual = boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(
                                                                                                    state, 
                                                                                                    endeff_frame_id, 
                                                                                                    Eigen::Vector3d::Zero()    
                                                                                                );
    
    Eigen::Vector3d lb = {-1.0, -1.0, -1.0};
    Eigen::Vector3d ub = {1.0, 0.4, 0.4};
    
    boost::shared_ptr<crocoddyl::ConstraintModelResidual> ee_constraint = boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                                                                                    state, 
                                                                                                    frameTranslationResidual,
                                                                                                    lb,
                                                                                                    ub
                                                                                                );

    // CREATING RUNNING MODELS
    std::vector< boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels;
    boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> terminal_model;
    const double dt = 5e-2;
    const int T = 40;

    for (unsigned t = 0; t < T + 1; ++t){
        boost::shared_ptr<crocoddyl::CostModelSum> runningCostModel = boost::make_shared<crocoddyl::CostModelSum>(state);
        runningCostModel->addCost("stateReg", xRegCost, 1e-4);
        // runningCostModel->addCost("ctrlRegGrav", uRegCost, 1e-4);
        if (t != T){
            runningCostModel->addCost("translation", frameTranslationCost, .1);
        }
        else{
            runningCostModel->addCost("translation", frameTranslationCost, .1);
        }
        boost::shared_ptr<crocoddyl::ConstraintModelManager> constraints = boost::make_shared<crocoddyl::ConstraintModelManager>(state, nu);    
        if(t != 0){
            constraints->addConstraint("ee_bound", ee_constraint);
        }

        // CREATING DAM MODEL
        boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> running_DAM = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
                                                                                                        state, 
                                                                                                        actuation,
                                                                                                        runningCostModel 
                                                                                                        // ,constraints
                                                                                                    );
        if (t != T){
            boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> running_model = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
                                                                                                        running_DAM,
                                                                                                        dt
                                                                                                        );
            runningModels.push_back(running_model);
        }
        else{
            terminal_model = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(running_DAM, dt);
        }
    }

    boost::shared_ptr<crocoddyl::ShootingProblem> problem = boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels, terminal_model); 


    #ifdef CROCODDYL_WITH_MULTITHREADING
    std::cout << "Crocoddyl was compiled with multithreading support." << std::endl;
    std::cout << "Number of threads: " << problem->get_nthreads() << std::endl;
    #else
    std::cout << "Crocoddyl was NOT compiled with multithreading support." << std::endl;
    return 0;
    #endif
    // Header
    
    std::cout << std::left << std::setw(42) << "      "
            << "  " << std::left << std::setw(15) << "AVG (ms)" << std::left
            << std::setw(15) << "STDDEV (ms)" << std::left << std::setw(15)
            << "MAX (ms)" << std::left << std::setw(15) << "MIN (ms)"
                << std::endl;


    // grg::Timer timer;

    // SETTING UP WARM START
    std::vector<Eigen::VectorXd> xs(T + 1, x0);
    std::vector<Eigen::VectorXd> us(T, Eigen::VectorXd::Zero(nu));
    // Unonstrained case
    // DEFINE SOLVER
    crocoddyl::SolverDDP solver_DDP = crocoddyl::SolverFDDP(problem);

    const int max_iter_SQP = 1;
    const int max_iter_GRG = 1;

    grg::SolverGRG solver_GRG = grg::SolverGRG(problem);
    grg::SolverGRG_ADAHESSIAN solver_GRG_ADAHESSIAN = grg::SolverGRG_ADAHESSIAN(problem);

    // crocoddyl::Stopwatch& swatch = crocoddyl::getProfiler();

    // swatch.enable_profiler();
    // std::cout<< "PROFILEING GRG" << std::endl;
    // swatch.set_mode(crocoddyl::REAL_TIME);
    solver_GRG.solve(xs, us, max_iter_GRG);
    solver_GRG_ADAHESSIAN.solve(xs, us, max_iter_GRG);
    // swatch.report_all(3);

    // std::cout << "\nPROFILEING DDP" << std::endl;
    // swatch.reset_all();
    // solver_DDP.solve(xs, us, max_iter_SQP);
    // swatch.report_all(3);

    


    // // solver_SQP.setCallbacks(false);
    

    // // SETTING UP STATISTICS
    // const int nb_DDP = 100;
    // const int nb_GRG = 100;
    // Eigen::VectorXd duration_DDP(nb_DDP);
    // Eigen::VectorXd duration_GRG(nb_GRG);
    // for (unsigned i = 0; i < nb_DDP; ++i){
    //     timer.start();
    //     solver_DDP.solve(xs, us, max_iter_SQP);
    //     timer.stop();
    //     duration_DDP[i] = timer.elapsed().user / max_iter_SQP;
    // }



    // for (unsigned i = 0; i < nb_GRG; ++i){
    //     timer.start();
    //     solver_GRG.solve(xs, us, max_iter_GRG);
    //     timer.stop();
    //     duration_GRG[i] = timer.elapsed().user / max_iter_GRG;
    // }


    // double const std_dev_DDP = std::sqrt((duration_DDP.array() - duration_DDP.mean()).square().sum() / (nb_DDP - 1));
    // double const std_dev_GRG = std::sqrt((duration_GRG.array() - duration_GRG.mean()).square().sum() / (nb_GRG - 1));

    // std::cout << "  " << std::left << std::setw(42) << "UR5 DDP" << std::left
    //           << std::setw(15) << duration_DDP.mean() << std::left << std::setw(15)
    //           << std_dev_DDP << std::left << std::setw(15)
    //           << duration_DDP.maxCoeff() << std::left << std::setw(15)
    //           << duration_DDP.minCoeff() << std::endl;

    // std::cout << "  " << std::left << std::setw(42) << "UR5 GRG" << std::left
    //           << std::setw(15) << duration_GRG.mean() << std::left << std::setw(15)
    //           << std_dev_GRG << std::left << std::setw(15)
    //           << duration_GRG.maxCoeff() << std::left << std::setw(15)
    //           << duration_GRG.minCoeff() << std::endl;


    // Constrained case 50
    // DEFINE SOLVER
    // mim_solvers::SolverCSQP solver_CSQP = mim_solvers::SolverCSQP(problem);
    // solver_CSQP.set_termination_tolerance(1e-4);
    // solver_CSQP.setCallbacks(false);
    // solver_CSQP.set_eps_abs(0.0);
    // solver_CSQP.set_eps_rel(0.0);
    // solver_CSQP.set_max_qp_iters(50);
    // const int max_iter_CSQP = 1;

    // // SETTING UP STATISTICS
    // const int nb_CSQP = 1000;
    // Eigen::VectorXd duration_CSQP(nb_CSQP);
    // for (unsigned i = 0; i < nb_CSQP; ++i){
    //     timer.start();
    //     solver_CSQP.solve(xs, us, max_iter_CSQP);
    //     timer.stop();
    //     duration_CSQP[i] = timer.elapsed().user;
    // }
    // double const std_dev_CSQP = std::sqrt((duration_CSQP.array() -  duration_CSQP.mean()).square().sum() / (nb_CSQP - 1));

    // std::cout << "  " << std::left << std::setw(42) << "UR5 CSQP (50 QP iters)" << std::left
    //           << std::setw(15) <<  duration_CSQP.mean() << std::left << std::setw(15)
    //           << std_dev_CSQP << std::left << std::setw(15)
    //           << duration_CSQP.maxCoeff() << std::left << std::setw(15)
    //           << duration_CSQP.minCoeff() << std::endl;

    // // Constrained case 200
    // solver_CSQP.set_max_qp_iters(200);

    // // SETTING UP STATISTICS
    // for (unsigned i = 0; i < nb_CSQP; ++i){
    //     timer.start();
    //     solver_CSQP.solve(xs, us, max_iter_CSQP);
    //     timer.stop();
    //     duration_CSQP[i] = timer.elapsed().user;
    // }
    // double const std_dev_CSQP1000 = std::sqrt((duration_CSQP.array() -  duration_CSQP.mean()).square().sum() / (nb_CSQP - 1));

    // std::cout << "  " << std::left << std::setw(42) << "UR5 CSQP (200 QP iters)" << std::left
    //           << std::setw(15) <<  duration_CSQP.mean() << std::left << std::setw(15)
    //           << std_dev_CSQP1000 << std::left << std::setw(15)
    //           << duration_CSQP.maxCoeff() << std::left << std::setw(15)
            //   << duration_CSQP.minCoeff() << std::endl;


    return 0;
}

