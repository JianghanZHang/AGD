#ifndef MIM_SOLVERS_GRG_HPP_
#define MIM_SOLVERS_GRG_HPP_

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#define EIGEN_USE_THREADS // Enable Eigen to use threading
#define EIGEN_USE_OPENMP // Specifically enable OpenMP support in Eigen
#include <Eigen/Dense> // Include Eigen headers after the macros
#endif  // CROCODDYL_WITH_MULTITHREADING

#include <Eigen/Cholesky>
#include <vector>
#include <boost/circular_buffer.hpp>
#include <vector>
#include <iostream>
#include <crocoddyl/core/solver-base.hpp>
#include <crocoddyl/core/mathbase.hpp>

namespace grg {

class SolverGRG : public crocoddyl::SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverGRG(boost::shared_ptr<crocoddyl::ShootingProblem> problem);
  virtual ~SolverGRG();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = crocoddyl::DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = crocoddyl::DEFAULT_VECTOR, 
                     const std::size_t maxiter = 100,
                     const bool is_feasible = false, const double reginit = 0.);

  virtual void forwardPass();
  /**
   * @brief Computes the merit function, gaps at the given xs, us along with delta x and delta u
   */

  virtual double calcDiff();

  virtual void computeDirection(const bool recalcDiff);

  virtual double tryStep(const double stepLength);

  virtual void checkKTConditions();
  
  virtual void backwardPass();

  virtual double stoppingCriteria();

  virtual const Eigen::Vector2d& expectedImprovement();

  virtual void resizeData();


  const std::vector<Eigen::VectorXd>& get_xs_try() const { return xs_try_; };
  const std::vector<Eigen::VectorXd>& get_us_try() const { return us_try_; };

  double get_KKT() const { return KKT_; };
  double get_gap_norm() const { return gap_norm_; };
  double get_xgrad_norm() const { return x_grad_norm_; };
  double get_ugrad_norm() const { return u_grad_norm_; };
  double get_merit() const { return merit_; };
  std::size_t get_filter_size() const { return filter_size_; };
  double get_mu() const { return mu_; };
  double get_beta1() const { return beta1_; };
  double get_beta2() const { return beta2_; };
  std::vector<Eigen::VectorXd> get_m() const { return m_; };
  std::vector<Eigen::VectorXd> get_v() const { return v_; };
  double get_cost() const { return cost_; };
  bool get_correct_bias() const { return correct_bias_; };
  double getConstStepLength() const { return const_step_length_; };
  std::vector<double> get_alphas() const { return alphas_; };
  bool get_use_line_search() const { return use_line_search_; };
  double get_termination_tolerance() const { return termination_tol_; };
  double get_line_search_parameter() const { return c_; };
  double get_curvature() const { return curvature_; };
  bool getMultithreading() const { return with_multithreading_; };

  void printCallbacks();
  void setCallbacks(bool inCallbacks);
  bool getCallbacks();

  void set_mu(double mu) { mu_ = mu; };
  void set_use_line_search(bool use_line_search) { use_line_search_ = use_line_search; };
  void set_beta1 (double beta1) { beta1_ = beta1; };
  void set_beta2 (double beta2) { beta2_ = beta2; };
  void set_m (std::vector<Eigen::VectorXd> m) { m_ = m; };
  void set_v (std::vector<Eigen::VectorXd> v) { v_ = v; };
  void set_correct_bias (bool correct_bias) { correct_bias_ = correct_bias; };
  void setConstStepLength (double const_step_length) { const_step_length_ = const_step_length; };
  void set_alphas (std::vector<double> alphas) { alphas_ = alphas; };
  void set_termination_tolerance(double tol) { termination_tol_ = tol; };
  void set_line_search_parameter(double c) { c_ = c; };


  
  
 public:
  std::vector<Eigen::VectorXd> xs_try_;  //!< State trajectory computed by line-search procedure
  std::vector<Eigen::VectorXd> us_try_;  //!< Control trajectory computed by line-search procedure

  std::vector<Eigen::VectorXd> tmpxs_;  //!< Control trajectory computed by line-search procedure

  double cost_try_;                      //!< Total cost computed by line-search procedure
  std::vector<Eigen::VectorXd> Vx_;   //!< Gradient of the Value function \f$\mathbf{V_x}\f$
  std::vector<Eigen::VectorXd> Vu_;   //!< Gradient of the Value function \f$\mathbf{V_u}\f$
  std::vector<Eigen::VectorXd> fs_try_;                        //!< Gaps/defects between shooting nodes
  std::vector<Eigen::VectorXd> dx_;                            //!< the descent direction for x
  std::vector<Eigen::VectorXd> du_;                            //!< the descent direction for u
  std::vector<Eigen::VectorXd> Vu_square_;                     //!< the squared Qu matrix
  std::vector<Eigen::VectorXd> m_;                     //!< the squared Qu matrix
  std::vector<Eigen::VectorXd> v_;                     //!< the squared Qu matrix
  std::vector<Eigen::VectorXd> m_corrected_;                     //!< the squared Qu matrix
  std::vector<Eigen::VectorXd> v_corrected_;                     //!< the squared Qu matrix

  Eigen::VectorXd xnext_;      //!< Next state \f$\mathbf{x}^{'}\f$

  std::vector<Eigen::VectorXd> lag_mul_;                       //!< the Lagrange multiplier of the dynamics constraint
  boost::circular_buffer<double> gap_list_;                    //!< memory buffer of gap norms (used in filter line-search)
  boost::circular_buffer<double> cost_list_;                   //!< memory buffer of gap norms (used in filter line-search)
  
 protected:
  double merit_ = 0;                                           //!< merit function at nominal traj
  double merit_try_ = 0;                                       //!< merit function for the step length tried
  double x_grad_norm_ = 0;                                     //!< 1 norm of the delta x
  double u_grad_norm_ = 0;                                     //!< 1 norm of the delta u
  double gap_norm_ = 0;                                        //!< 1 norm of the gaps
  double gap_norm_try_ = 0;                                    //!< 1 norm of the gaps
  double cost_ = 0;                                            //!< cost function
  bool with_callbacks_ = false;                                //!< With callbacks
  std::size_t filter_size_ = 1;                                //!< Filter size for line-search (do not change the default value !)
  double mu_ = 0.;                                            //!< penalty no constraint violation
  double beta1_ = 0.9;
  double beta2_ = 0.999;
  bool correct_bias_ = true;
  bool use_line_search_ = false;
  double const_step_length_ = 0.01;
  double KKT_ = std::numeric_limits<double>::infinity();   //!< KKT conditions residual
  double ub_ = std::numeric_limits<double>::infinity();    //!< Upper bound for Goldenstein line-search
  double lb_ = 0;                                          //!< Lower bound for Goldenstein line-search
  double c_ = 0.25;                                         //!< Goldenstein line-search parameter
  std::vector<double> alphas_;                             //!< Set of step lengths using by the line-search procedure
  double curvature_ = 0;                                   //!< Curvature of the cost function
  double tmp_ub_ = 0;                                    //!< Temporary variable                        
  double tmp_lb_ = 0;                                         //!< Temporary variable

  double control_ub_ = 100;                                    //!< Control upper bound
  double control_lb_ = -100;                                   //!< Control lower bound
  double state_ub_ = 4;                                      //!< State upper bound
  double state_lb_ = -4;                                     //!< State lower bound
  int fail_time_tmp_ = 0;                                     //!< Temporary variable
  bool with_multithreading_ = false;                            //!< Boolean for multi-threading
  
 private:
  double th_acceptnegstep_;                                    //!< Threshold used for accepting step along ascent direction
  bool is_worse_than_memory_ = false;                          //!< Boolean for filter line-search criteria 
  Eigen::VectorXd tmp_vec_x_;                                  //!< Temporary variable
  std::vector<Eigen::VectorXd> tmp_vec_u_;                     //!< Temporary variable
  Eigen::VectorXd fs_flat_;                                //!< Gaps/defects between shooting nodes (1D array)

  double termination_tol_ = 1e-6;                          //!< Termination tolerance
};

}  // namespace mim_solvers

#endif  // MIM_SOLVERS_SQP_HPP_
