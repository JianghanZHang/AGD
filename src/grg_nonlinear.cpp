///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>

#include <crocoddyl/core/utils/exception.hpp>
#include "grg/grg_nonlinear.hpp"

using namespace crocoddyl;

namespace grg {

SolverGRG_nonlinear::SolverGRG_nonlinear(boost::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverAbstract(problem){

      if (!problem_) {
        std::cerr << "SolverGRG_nonlinear constructor received a nullptr for problem." << std::endl;
        throw std::invalid_argument("SolverGRG_nonlinear constructor requires a non-null problem");
        }
        
      #ifdef CROCODDYL_WITH_MULTITHREADING
      with_multithreading_ = true;
      #endif  // CROCODDYL_WITH_MULTITHREADING

      const std::size_t T = problem_->get_T();
      const std::size_t ndx = problem_->get_ndx();
      xs_try_.resize(T + 1);
      us_try_.resize(T);
      fs_try_.resize(T + 1);
      dx_.resize(T+1);
      du_.resize(T);
      Vx_.resize(T + 1);
      Vu_.resize(T);
      Vu_square_.resize(T);
      m_corrected_.resize(T);
      v_corrected_.resize(T);
      m_.resize(T);
      v_.resize(T);
      fs_flat_.resize(ndx*(T + 1));
      fs_flat_.setZero();
      gap_list_.resize(filter_size_);
      cost_list_.resize(filter_size_);
      tmpxs_.resize(T + 1);
      const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
      for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
        const std::size_t nu = model->get_nu();

        Vx_[t] = Eigen::VectorXd::Zero(ndx);
        Vu_[t] = Eigen::VectorXd::Zero(nu);
        m_corrected_[t] = Eigen::VectorXd::Zero(nu);
        v_corrected_[t] = Eigen::VectorXd::Zero(nu);
        m_[t] = Eigen::VectorXd::Zero(nu);
        v_[t] = Eigen::VectorXd::Zero(nu);

        if (t == 0) {
            xs_try_[t] = problem_->get_x0();
        } 
        else {
            xs_try_[t] = model->get_state()->zero();
        }
        us_try_[t] = Eigen::VectorXd::Zero(nu);
        dx_[t] = Eigen::VectorXd::Zero(ndx);
        du_[t] = Eigen::VectorXd::Zero(nu);

        fs_try_[t] = Eigen::VectorXd::Zero(ndx);
        du_[t] = Eigen::VectorXd::Zero(nu);
        tmpxs_[t] = Eigen::VectorXd::Zero(ndx);
      }
      dx_.back() = Eigen::VectorXd::Zero(ndx);

      fs_try_.back() = Eigen::VectorXd::Zero(ndx);
      xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();
      Vx_.back() = Eigen::VectorXd::Zero(ndx);
      
      const std::size_t n_alphas = 10;
      alphas_.resize(n_alphas);
      for (std::size_t n = 0; n < n_alphas; ++n) {
        alphas_[n] = 2. / pow(2., static_cast<double>(n));
      }
    }

SolverGRG_nonlinear::~SolverGRG_nonlinear() {}

bool SolverGRG_nonlinear::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                       const std::size_t maxiter, const bool is_feasible, const double reginit) {
  const std::size_t T = problem_->get_T();

  START_PROFILER("SolverGRG_nonlinear::solve");
  (void)is_feasible;
  
  if (problem_->is_updated()) {
    resizeData();
  }
  fail_time_tmp_ = 0;
  tmpxs_ = problem_->rollout_us(init_us);
  setCandidate(tmpxs_, init_us, false);

  xs_[0] = problem_->get_x0();      // Otherwise xs[0] is overwritten by init_xs inside setCandidate()
  xs_try_[0] = problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible

  bool recalcDiff = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {

    recalcDiff = true;

    while (true) {
      try {
        computeDirection(recalcDiff);
      } 
      catch (std::exception& e) {
        return false;
      }
      break;
    }

    // KKT termination criteria
    checkKTConditions();
    
    if (KKT_ <= termination_tol_) {
      if(with_callbacks_){
        printCallbacks();
      }
      STOP_PROFILER("SolverGRG_nonlinear::solve");
      return true;
    }
      
    cost_list_.push_back(cost_);

    if(use_line_search_){
        // We need to recalculate the derivatives when the step length passes
        tmp_ub_ = c_ * curvature_;
        tmp_lb_ = (1-c_) * curvature_;
        for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
        steplength_ = *it;
        try {
            tryStep(steplength_);
            ub_ = merit_+ steplength_ * tmp_ub_;
            lb_ = merit_ + steplength_ * tmp_lb_;
        } catch (std::exception& e) {
            continue;
        }
        
        // filter line search criteria
        if ((merit_try_ <= merit_ ) || (steplength_ == alphas_.back())){
            setCandidate(xs_try_, us_try_, false);
            recalcDiff = true;
            if(steplength_ == alphas_.back()){
              fail_time_tmp_ ++;
            }
            break;
          }
        }
    }
    else{
        steplength_ = const_step_length_;
        tryStep(steplength_);
        setCandidate(xs_try_, us_try_, true);
        recalcDiff = true;
    }
    if(with_callbacks_){
      printCallbacks();
    }
  }
  STOP_PROFILER("SolverGRG_nonlinear::solve");

  return false;

}

void SolverGRG_nonlinear::computeDirection(const bool recalcDiff){
  START_PROFILER("SolverGRG_nonlinear::computeDirection");
  if (recalcDiff) {
      calcDiff();
  }
  const std::size_t T = problem_->get_T();

  backwardPass();

  STOP_PROFILER("SolverGRG_nonlinear::computeDirection");
}

void SolverGRG_nonlinear::checkKTConditions(){
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

double SolverGRG_nonlinear::stoppingCriteria() {
  stop_ = 0.;
  const std::size_t T = this->problem_->get_T();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();

  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t nu = models[t]->get_nu();
    if (nu != 0) {
      stop_ += Vu_[t].squaredNorm();
    }
  }
  return stop_;
}

const Eigen::Vector2d& SolverGRG_nonlinear::expectedImprovement() {
  d_.fill(0.0);

  return d_;
}

void SolverGRG_nonlinear::resizeData() {
  START_PROFILER("SolverGRSolverGRG_nonlinearG::resizeData");
  SolverAbstract::resizeData();

  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();

    Vx_[t].conservativeResize(ndx);
    Vu_[t].conservativeResize(nu);

    m_[t].conservativeResize(nu);
    v_[t].conservativeResize(nu);
  }
  STOP_PROFILER("SolverGRG_nonlinear::resizeData");
}

double SolverGRG_nonlinear::calcDiff() {
  START_PROFILER("SolverGRG_nonlinear::calcDiff");
  if (iter_ == 0) {
    problem_->calc(xs_, us_);
  }
  cost_ = problem_->calcDiff(xs_, us_);

  ffeas_ = computeDynamicFeasibility();
  STOP_PROFILER("SolverGRG_nonlinear::calcDiff");
  return cost_;
}

void SolverGRG_nonlinear::backwardPass() {
  START_PROFILER("SolverGRG_nonlinear::backwardPass");
  const boost::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  Vx_.back() = d_T->Lx;

  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1];
    const std::size_t nu = m->get_nu();

    START_PROFILER("SolverGRG_nonlinear::Qx");
    Vx_[t] = d->Lx;
    Vx_[t].noalias() += d->Fx.transpose() * Vx_p;
    STOP_PROFILER("SolverGRG_nonlinear::Qx");
    if (nu != 0) {
      START_PROFILER("SolverGRG_nonlinear::Qu");
      Vu_[t] = d->Lu;
      Vu_[t].noalias() += d->Fu.transpose() * Vx_p;
      STOP_PROFILER("SolverGRG_nonlinear::Qu");

    }

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
  }
  STOP_PROFILER("SolverGRG_nonlinear::backwardPass");
}


void SolverGRG_nonlinear::forwardPass(const double steplength){
    START_PROFILER("SolverGRG_nonlinear::forwardPass");
    x_grad_norm_ = 0; 
    u_grad_norm_ = 0;

    const std::size_t T = problem_->get_T();
    const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
    const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();

    if(correct_bias_){
        #ifdef CROCODDYL_WITH_MULTITHREADING
        #pragma omp parallel for num_threads(problem_->get_nthreads())
        #endif
        for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
        Vu_square_[t] = Vu_[t].cwiseProduct(Vu_[t]);
        m_[t] = beta1_ * m_[t] + (1 - beta1_) * Vu_[t];
        v_[t] = beta2_ * v_[t] + (1 - beta2_) * Vu_square_[t];

        m_corrected_[t] = m_[t] / (1 - pow(beta1_, iter_+2));
        v_corrected_[t] = v_[t] / (1 - pow(beta2_, iter_+2));

        du_[t] = - m_corrected_[t].cwiseQuotient(v_corrected_[t].cwiseSqrt());
        u_grad_norm_ += du_[t].lpNorm<1>();
        us_try_[t].noalias() = us_[t];
        us_try_[t].noalias() += du_[t] * steplength; 
        }
        
        for (std::size_t t = 0; t < T; ++t) {
          const boost::shared_ptr<ActionModelAbstract>& m = models[t];
          const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
          if (m->get_nu() != 0) {
            
            m->calc(d, xs_try_[t], us_try_[t]);
          } else {
            m->calc(d, xs_try_[t]);
          }
          xs_try_[t + 1] = d->xnext;
          cost_try_ += d->cost;

          if (raiseIfNaN(cost_try_)) {
          STOP_PROFILER("SolverGRG_nonlinear::tryStep");
          throw_pretty("step_error");
        }   
        }
    }

    else{

        #ifdef CROCODDYL_WITH_MULTITHREADING
        #pragma omp parallel for num_threads(problem_->get_nthreads())
        #endif
        for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
        Vu_square_[t] = Vu_[t].cwiseProduct(Vu_[t]);
        m_[t] = (1 - beta1_) * Vu_[t] + beta1_ * m_[t];
        v_[t] = (1 - beta2_) * Vu_square_[t] + beta2_ * v_[t];
        
        du_[t] = -m_[t].cwiseQuotient(v_[t].cwiseSqrt());
        u_grad_norm_ += du_[t].lpNorm<1>();
        us_try_[t].noalias() = us_[t];
        us_try_[t].noalias() += du_[t] * steplength; 
        }
        
        for (std::size_t t = 0; t < T; ++t) {
          const boost::shared_ptr<ActionModelAbstract>& m = models[t];
          const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
          if (m->get_nu() != 0) {
            m->calc(d, xs_try_[t], us_try_[t]);
          } else {
            m->calc(d, xs_try_[t]);
          }
          xs_try_[t + 1] = d->xnext;
          cost_try_ += d->cost;

          if (raiseIfNaN(cost_try_)) {
          STOP_PROFILER("SolverGRG_nonlinear::tryStep");
          throw_pretty("step_error");
        }   
        }
        
    }

      const boost::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
      const boost::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
      m->calc(d, xs_try_.back());
      cost_try_ += d->cost;

    u_grad_norm_ = u_grad_norm_/(double)T; 
    STOP_PROFILER("SolverGRG_nonlinear::forwardPass");

}

double SolverGRG_nonlinear::tryStep(const double steplength) {
    // if (steplength > 1. || steplength < 0.) {
    //     throw_pretty("Invalid argument: "
    //                 << "invalid step length, value is between 0. to 1.");
    // }
    if (steplength < 0.) {
        throw_pretty("Invalid argument: "
                    << "invalid step length, value is between 0. to 1.");
    }
    START_PROFILER("SolverGRG_nonlinear::tryStep");
    cost_try_ = 0.;
    forwardPass(steplength);
    STOP_PROFILER("SolverGRG_nonlinear::tryStep");

    return cost_try_;
}

void SolverGRG_nonlinear::printCallbacks(){
  if (this->get_iter() % 10 == 0) {
    std::cout << "iter     merit         cost         grad      step    ||gaps||        KKT        merit_try        cost_try";
    std::cout << std::endl;
  }
  if(KKT_ < termination_tol_){
    std::cout << std::setw(4) << "END" << "  ";
  } else {
    std::cout << std::setw(4) << this->get_iter() << "  ";
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
  std::cout << std::scientific << std::setprecision(5) << KKT_ << "  ";
  std::cout << std::scientific << std::setprecision(5) << merit_try_ << "  ";
  std::cout << std::scientific << std::setprecision(5) << cost_try_ << "  ";
  std::cout << std::endl;
  // std::cout << "line search fail time: " << fail_time_tmp_;
  std::cout << std::endl;
  std::cout << std::flush;
}

void SolverGRG_nonlinear::setCallbacks(bool inCallbacks){
  with_callbacks_ = inCallbacks;
}

bool SolverGRG_nonlinear::getCallbacks(){
  return with_callbacks_;
}
  
}  // namespace grg
