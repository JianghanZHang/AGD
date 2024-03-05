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
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"


#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"

#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"


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

    const int endeff_frame_id = rmodel->getFrameId("ee_link");
    Eigen::Vector3d endeff_translation = {0.3, 0.3, 0.3};

    boost::shared_ptr<crocoddyl::ResidualModelFramePlacement> framePlacementResidual = boost::make_shared<crocoddyl::ResidualModelFramePlacement>(
                                                                                                    state, 
                                                                                                    endeff_frame_id, 
                                                                                                    pinocchio::SE3(Eigen::Matrix3d::Identity(), endeff_translation)    
                                                                                                );

    boost::shared_ptr<crocoddyl::ResidualModelFrameTranslation> frameTranslationResidual = boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(
                                                                                                    state, 
                                                                                                    endeff_frame_id, 
                                                                                                    endeff_translation    
                                                                                                );

    boost::shared_ptr<crocoddyl::CostModelResidual> frameTranslationCost = boost::make_shared<crocoddyl::CostModelResidual>(state, frameTranslationResidual); 
    boost::shared_ptr<crocoddyl::CostModelResidual> framePlacementCost = boost::make_shared<crocoddyl::CostModelResidual>(state, framePlacementResidual); 

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
    const double dt = 1e-2;
    const int T = 51;

    for (unsigned t = 0; t < T + 1; ++t){
        boost::shared_ptr<crocoddyl::CostModelSum> runningCostModel = boost::make_shared<crocoddyl::CostModelSum>(state);
        if (t != T){
            runningCostModel->addCost("stateReg", xRegCost, 1e-1);
            runningCostModel->addCost("ctrlRegGrav", uRegCost, 1e-4);
            runningCostModel->addCost("translation", frameTranslationCost, 10.);
            // runningCostModel->addCost("placement", framePlacementCost, 10.);
        }
        else{
            runningCostModel->addCost("translation", frameTranslationCost, 10.);
            runningCostModel->addCost("stateReg", xRegCost, 1e-1);
            // runningCostModel->addCost("placement", framePlacementCost, 10.);
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
            terminal_model = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(running_DAM, 0.);
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
    std::cout << "T:" << problem->get_T() << std::endl;
    // SETTING UP WARM START
    std::vector<Eigen::VectorXd> xs(T, x0);
    std::vector<Eigen::VectorXd> us(T, Eigen::VectorXd::Zero(nu));
    // problem->quasiStatic(us, xs);
    us = problem->quasiStatic_xs(xs);
    xs.resize(T+1);
    xs = problem->rollout_us(us);
    std::cout << "init_us[0]:" << us[0] << std::endl;
    // us.resize(T);
    // us.push_back(us[0]);
    // Unonstrained case
    // DEFINE SOLVER

    grg::SolverGRG_ADAHESSIAN solver_GRG = grg::SolverGRG_ADAHESSIAN(problem);
    crocoddyl::SolverDDP solver_DDP(problem);
    solver_GRG.set_const_step_length(.2);
    // solver_SQP.setCallbacks(false);
    solver_GRG.setCallbacks(true);
    const int max_iter_GRG = 20000;
    const int max_iter_DDP = 1000;

    solver_GRG.solve(xs, us, max_iter_GRG);
    std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
    cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
    solver_DDP.setCallbacks(cbs);
    solver_DDP.solve(xs, us, max_iter_DDP);


    return 0;
}

