///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif  // CROCODDYL_WITH_MULTITHREADING

#include <iostream>
#include <iomanip>

#include <crocoddyl/core/utils/exception.hpp>
#include "grg/grg.hpp"
#include "grg.hpp"

using namespace crocoddyl;

namespace grg {

SolverGRG::SolverGRG(boost::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverAbstract(problem){
      
      const std::size_t T = this->problem_->get_T();
      const std::size_t ndx = problem_->get_ndx();
      fs_try_.resize(T + 1);
      dx_.resize(T+1);
      du_.resize(T);
      Vu_square_.resize(T);
      m_.resize(T);
      v_.resize(T);
      gap_list_.resize(filter_size_);
      cost_list_.resize(filter_size_);
      const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
      for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
        const std::size_t nu = model->get_nu();
        dx_[t].resize(ndx); du_[t].resize(nu);
        fs_try_[t].resize(ndx);
        dx_[t].setZero();
        du_[t] = Eigen::VectorXd::Zero(nu);
        fs_try_[t] = Eigen::VectorXd::Zero(ndx);
        tmp_vec_u_[t].resize(nu);
      }
      dx_.back().resize(ndx);
      dx_.back().setZero();
      fs_try_.back().resize(ndx);
      fs_try_.back() = Eigen::VectorXd::Zero(ndx);

      const std::size_t n_alphas = 10;
      alphas_.resize(n_alphas);
      for (std::size_t n = 0; n < n_alphas; ++n) {
        alphas_[n] = 1. / pow(2., static_cast<double>(n));
      }
    }

SolverGRG::~SolverGRG() {}

bool SolverGRG::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                       const std::size_t maxiter, const bool is_feasible) {

  START_PROFILER("SolverGRG::solve");
  (void)is_feasible;
  
  if (problem_->is_updated()) {
    resizeData();
  }
  setCandidate(init_xs, init_us, false);
  xs_[0] = problem_->get_x0();      // Otherwise xs[0] is overwritten by init_xs inside setCandidate()
  xs_try_[0] = problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible

  bool recalcDiff = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {

    recalcDiff = true;

    while (true) {
      try {
        computeDirection(recalcDiff, iter_);
      } 
      catch (std::exception& e) {
        return false;
      }
      break;
    }

    // KKT termination criteria
    checkKKTConditions();
    if (KKT_ <= termination_tol_) {
      if(with_callbacks_){
        printCallbacks();
      }
      STOP_PROFILER("SolverGRG::solve");
      return true;
    }
      
    gap_list_.push_back(gap_norm_);
    cost_list_.push_back(cost_);

    if(ust_line_search_){
        // We need to recalculate the derivatives when the step length passes
        for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
        steplength_ = *it;
        try {
            merit_try_ = tryStep(steplength_);
        } catch (std::exception& e) {
            continue;
        }
        // Filter line search criteria 
        // Equivalent to heuristic cost_ > cost_try_ || gap_norm_ > gap_norm_try_ when filter_size=1
        if(use_filter_line_search_){
            is_worse_than_memory_ = false;
            std::size_t count = 0.; 
            while( count < filter_size_ && is_worse_than_memory_ == false and count <= iter_){
            is_worse_than_memory_ = cost_list_[filter_size_-1-count] <= cost_try_ && gap_list_[filter_size_-1-count] <= gap_norm_try_;
            count++;
            }
            if( is_worse_than_memory_ == false ) {
            setCandidate(xs_try_, us_try_, false);
            recalcDiff = true;
            break;
            } 
        }
        // Line-search criteria using merit function
        else{
            if (merit_ > merit_try_) {
            setCandidate(xs_try_, us_try_, false);
            recalcDiff = true;
            break;
            }
        }
        }
    }
    else{
        merit_try_ = tryStep(const_step_length_);
        setCandidate(xs_try_, us_try_, false);
        recalcDiff = true;
    }

    // if (steplength_ > th_stepdec_) {
    //   decreaseRegularization();
    // }
    // if (steplength_ <= th_stepinc_) {
    //   increaseRegularization();
    //   if (preg_ == reg_max_) {
    //     STOP_PROFILER("SolverGRG::solve");
    //     return false;
    //   }
    // }

    if(with_callbacks_){
      printCallbacks();
    }
  }

}

void SolverGRG::computeDirection(const bool recalcDiff, std::size_t iter_){
  START_PROFILER("SolverGRG::computeDirection");
  if (recalcDiff) {
    cost_ = calcDiff();
  }
  gap_norm_ = 0;
  const std::size_t T = problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    gap_norm_ += fs_[t].lpNorm<1>();   
  }
  gap_norm_ += fs_.back().lpNorm<1>();   

  merit_ = cost_ + mu_*gap_norm_;

  backwardPass();
  forwardPass(iter_);

  STOP_PROFILER("SolverGRG::computeDirection");
}

void SolverGRG::checkKKTConditions(){
  KKT_ = 0.;
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
    
    KKT_ = std::max(KKT_, Vx_[t].lpNorm<Eigen::Infinity>());
    KKT_ = std::max(KKT_, Vu_[t].lpNorm<Eigen::Infinity>());
    fs_flat_.segment(t*ndx, ndx) = fs_[t];
  }
  fs_flat_.tail(ndx) = fs_.back();
  const boost::shared_ptr<ActionDataAbstract>& d_ter = problem_->get_terminalData();
  KKT_ = std::max(KKT_, Vx_[T].lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
}


double SolverGRG::calcDiff() {
  START_PROFILER("SolverGRG::calcDiff");
  if (iter_ == 0) {
    problem_->calc(xs_, us_);
  }
  cost_ = problem_->calcDiff(xs_, us_);

  ffeas_ = computeDynamicFeasibility();
  STOP_PROFILER("SolverGRG::calcDiff");
  return cost_;
}

void SolverGRG::backwardPass() {
  START_PROFILER("SolverGRG::backwardPass");
  const boost::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  Vx_.back() = d_T->Lx;

  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1];
    const std::size_t nu = m->get_nu();

    START_PROFILER("SolverGRG::Qx");
    Vx_[t] = d->Lx;
    Vx_[t].noalias() += d->Fx.transpose() * Vx_p;
    STOP_PROFILER("SolverGRG::Qx");
    if (nu != 0) {
      START_PROFILER("SolverGRG::Qu");
      Vu_[t] = d->Lu;
      Vu_[t].noalias() += d->Fu.transpose() * Vx_p;
      STOP_PROFILER("SolverGRG::Qu");

    }

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
  }
  STOP_PROFILER("SolverGRG::backwardPass");
}


void SolverGRG::forwardPass(std::size_t iter_){
    START_PROFILER("SolverGRG::forwardPass");
    x_grad_norm_ = 0; 
    u_grad_norm_ = 0;

    const std::size_t T = problem_->get_T();
    const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();

    if(correct_bias_){
    
        for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
        Vu_square_[t] = Vu_[t].cwiseProduct(Vu_[t]);
        m_[t] = (1 - beta1_) * Vu_[t] + beta1_ * m_[t];
        v_[t] = (1 - beta2_) * Vu_square_[t] + beta2_ * v_[t];

        m_corrected_[t] = m_[t] / (1 - pow(beta1_, iter_+2));
        v_corrected_[t] = v_[t] / (1 - pow(beta2_, iter_+2));

        du_[t].noalias() = - m_corrected_[t].cwiseQuotient(v_corrected_[t].cwiseSqrt());

        dx_[t+1].noalias() = fs_[t+1];
        dx_[t+1].noalias() += d->Fu * du_[t];
        dx_[t+1].noalias() += d->Fx * dx_[t];
        x_grad_norm_ += dx_[t].lpNorm<1>(); // assuming that there is no gap in the initial state
        u_grad_norm_ += du_[t].lpNorm<1>();
        }

    }

    else{

        for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
        Vu_square_[t] = Vu_[t].cwiseProduct(Vu_[t]);
        m_[t] = (1 - beta1_) * Vu_[t] + beta1_ * m_[t];
        v_[t] = (1 - beta2_) * Vu_square_[t] + beta2_ * v_[t];

        du_[t].noalias() = -m_[t].cwiseQuotient(v_[t].cwiseSqrt());

        dx_[t+1].noalias() = fs_[t+1];
        dx_[t+1].noalias() += d->Fu * du_[t];
        dx_[t+1].noalias() += d->Fx * dx_[t];
        x_grad_norm_ += dx_[t].lpNorm<1>(); // assuming that there is no gap in the initial state
        u_grad_norm_ += du_[t].lpNorm<1>();
        }

    }

    x_grad_norm_ += dx_.back().lpNorm<1>(); // assuming that there is no gap in the initial state
    x_grad_norm_ = x_grad_norm_/(double)(T+1);
    u_grad_norm_ = u_grad_norm_/(double)T; 
    STOP_PROFILER("SolverGRG::forwardPass");

}

double SolverGRG::tryStep(const double steplength) {
    if (steplength > 1. || steplength < 0.) {
        throw_pretty("Invalid argument: "
                    << "invalid step length, value is between 0. to 1.");
    }
    START_PROFILER("SolverGRG::tryStep");
    cost_try_ = 0.;
    merit_try_ = 0;
    gap_norm_try_ = 0;

    const std::size_t T = problem_->get_T();
    const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
    const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();

    for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<ActionModelAbstract>& m = models[t];
        const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
        const std::size_t nu = m->get_nu();

        m->get_state()->integrate(xs_[t], steplength * dx_[t], xs_try_[t]); 
        if (nu != 0) {
            us_try_[t].noalias() = us_[t];
            us_try_[t].noalias() += steplength * du_[t];
        }        
        m->calc(d, xs_try_[t], us_try_[t]);        
        cost_try_ += d->cost;

        if (t > 0){
          const boost::shared_ptr<ActionDataAbstract>& d_prev = datas[t-1];
          m->get_state()->diff(xs_try_[t], d_prev->xnext, fs_try_[t-1]);
          gap_norm_try_ += fs_try_[t-1].lpNorm<1>(); 
        } 

        if (raiseIfNaN(cost_try_)) {
          STOP_PROFILER("SolverGRG::tryStep");
          throw_pretty("step_error");
        }   
    }

    // Terminal state update
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m_ter = problem_->get_terminalModel();
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_ter = problem_->get_terminalData();
    m_ter->get_state()->integrate(xs_.back(), steplength * dx_.back(), xs_try_.back()); 
    m_ter->calc(d_ter, xs_try_.back());
    cost_try_ += d_ter->cost;

    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[T-1];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[T-1];
    
    m->get_state()->diff(xs_try_.back(), d->xnext, fs_try_[T-1]);
    gap_norm_try_ += fs_try_[T-1].lpNorm<1>(); 

    merit_try_ = cost_try_ + mu_*gap_norm_try_;

    if (raiseIfNaN(cost_try_)) {
        STOP_PROFILER("SolverGRG::tryStep");
        throw_pretty("step_error");
    }

    STOP_PROFILER("SolverGRG::tryStep");

    return merit_try_;
}

void SolverGRG::printCallbacks(){
  if (this->get_iter() % 10 == 0) {
    std::cout << "iter     merit         cost         grad      step    ||gaps||        KKT";
    std::cout << std::endl;
  }
  if(KKT_ < termination_tol_){
    std::cout << std::setw(4) << "END" << "  ";
  } else {
    std::cout << std::setw(4) << this->get_iter()+1 << "  ";
  }
  std::cout << std::scientific << std::setprecision(5) << this->get_merit() << "  ";
  std::cout << std::scientific << std::setprecision(5) << this->get_cost() << "  ";
  std::cout << this->get_xgrad_norm() + this->get_ugrad_norm() << "  ";
  if(KKT_ < termination_tol_){
    std::cout << std::fixed << std::setprecision(4) << " ---- " << "  ";
  } else {
    std::cout << std::fixed << std::setprecision(4) << this->get_steplength() << "  ";
  }
  std::cout << std::scientific << std::setprecision(5) << this->get_gap_norm() << "  ";
  std::cout << std::scientific << std::setprecision(5) << KKT_;
  std::cout << std::endl;
  std::cout << std::flush;
}

void SolverGRG::setCallbacks(bool inCallbacks){
  with_callbacks_ = inCallbacks;
}

bool SolverGRG::getCallbacks(){
  return with_callbacks_;
}
// double SolverSQP::get_th_acceptnegstep() const { return th_acceptnegstep_; }

// void SolverSQP::set_th_acceptnegstep(const double th_acceptnegstep) {
//   if (0. > th_acceptnegstep) {
//     throw_pretty("Invalid argument: "
//                  << "th_acceptnegstep value has to be positive.");
//   }
//   th_acceptnegstep_ = th_acceptnegstep;
// }

}  // namespace mim_solvers
