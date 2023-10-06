#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

// 포인트의 time이 반대인지 확인
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;    // IMU 파라미터 출력 파일
  V3D cov_acc;          // 가속도 측정값 공분산
  V3D cov_gyr;          // 각속도 측정값 공분산
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;     // 각속도 측정값 공분산 bias
  V3D cov_bias_acc;     // 가속도 측정값 공분산 bias
  double first_lidar_time;  // 현재 프레임의 첫 번째 포인트 클라우드 시간

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;    // 현재 프레임 포인트 클라우드는 왜곡되어있지 않음
  sensor_msgs::ImuConstPtr last_imu_; // 이전 프레임에서 imu의 마지막 값
  deque<sensor_msgs::ImuConstPtr> v_imu_; // IMU queue
  vector<Pose6D> IMUpose;                 // IMU pose
  vector<M3D>    v_rot_pcl_;              // 사용되지 않음
  M3D Lidar_R_wrt_IMU;            // LiDAR에서 IMU로의 external parameter : Rotation
  V3D Lidar_T_wrt_IMU;            // LiDAR에서 IMU로의 external parameter : position
  V3D mean_acc;     // 가속도 평균. 분산을 계산할 때 사용
  V3D mean_gyr;     // 각속도 평균. 분산을 계산할 때 사용
  V3D angvel_last;    // 이전 프레임의 각속도
  V3D acc_s_last;     // 이전 프레임의 가속도
  double start_timestamp_;      // 시작 타임스탬프
  double last_lidar_end_time_;  // 마지막 프레임 종료 타임스탬프
  int    init_iter_num = 1;     // iteration 수 초기화
  bool   b_first_frame_ = true;   // first frame 이면 true
  bool   imu_need_init_ = true;   // IMU 초기화가 필요하면 true
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;        // iteration 수 초기화
  Q = process_noise_cov();  // use-ikfom.hpp 의 process_noise_cov() : 노이즈 공분산의 초기화
  cov_acc       = V3D(0.1, 0.1, 0.1);   // 가속도 측정값 공분산 초기화
  cov_gyr       = V3D(0.1, 0.1, 0.1);   // 각속도 측정값 공분산 초기화
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);  // 각속도 측정값 공분산 bias 초기화 
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);  // 가속도 측정값 공분산 bias 초기화
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;   // 이전 프레임의 각속도 초기화
  Lidar_T_wrt_IMU = Zero3d;   // LiDAR에서 IMU로의 external parameter 초기화 : position
  Lidar_R_wrt_IMU = Eye3d;    // LiDAR에서 IMU로의 external parameter 초기화 : Rotation
  last_imu_.reset(new sensor_msgs::Imu());  // 이전 프레임 IMU 초기화
}

ImuProcess::~ImuProcess() {}

// 파라미터 리셋
void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;   // imu init이 필요하다
  start_timestamp_  = -1;     // 시작 타임스탬프
  init_iter_num     = 1;      // iteration 수 초기화
  v_imu_.clear();             // IMU queue clear
  IMUpose.clear();            // IMU pose clear
  last_imu_.reset(new sensor_msgs::Imu());  // 이전 프레임 IMU 초기화
  cur_pcl_un_.reset(new PointCloudXYZI());  // 현재 프레임 포인트 클라우드(왜곡X) 초기화
}

// external 파라미터 R,T 설정 : transformation mtx 들어온 경우
void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

// external 파라미터 R(I),T 설정 : translation 만 들어온 경우
void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

// external 파라미터 R,T 설정 : translation, rotation 들어온 경우
void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

// 자이로스코프 각속도 공분산 설정
void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

// accelerometer 가속도 공분산 설정
void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

// 자이로스코프 각속도 공분산 bias 설정
void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

// accelerometer 가속도 공분산 bias 설정
void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  // static initialization
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_) // 만약 first 프레임 이라면
  {
    Reset();  // 파라미터 리셋
    N = 1;    // iteration 카운트를 1로 설정
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;  // common_lib.h : initial 모멘트의 imu 가속도 가져오기
    const auto &gyr_acc = meas.imu.front()->angular_velocity;     // common_lib.h : initial 모멘트의 imu 각속도 가져오기
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;    // 가속도 측정값을 초기 평균으로
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;    // 각속도 측정값을 초기 평균으로
    first_lidar_time = meas.lidar_beg_time;   // 현재 IMU 프레임에 해당하는 라이다 time을 initial time으로 사용
  }

  // 분산 계산
  for (const auto &imu : meas.imu)  // 모든 IMU 프레임에 대해
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    // 현재 프레임에 기반해 평균과 mean difference 업데이트 : moving average
    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    // .cwiseProduct() : 해당하는 계수들을 곱한다.
    // 각 iteration 이후 평균이 변하고, 최종 분산 공식에서 최종 평균을 마이너스 연산
    // variance recursion fomula (wrong fomula! 위에서 mean_acc, mean_gyr 업데이트 안 했으면 맞는 공식이지만, 했으므로 틀림. 여기서 구한 값 사용하지 않고 나중에 scalar로 사용.)
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  state_ikfom init_state = kf_state.get_x();  // esekfom.hpp에서 x_ status 가져오기
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);  // 중력(common_lib.h)과 가속도 측정값 평균의 단위 중력을 사용해 회전 행렬(SO2)의 중력 가속도 계산
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;                  // 각속도 측정값을 자이로스코프 bias로
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;  // external parameter T
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;  // external parameter R
  kf_state.change_x(init_state);              // initializaiton status를 esekfom.hpp의 x_에 전달

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();  // esekfom.hpp에서 공분산 행렬 P_ 가져오기
  init_P.setIdentity();                                                         // 공분산 행렬을 I로 설정
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;                            // position, rotation 공분산 설정
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;                        // 속도와 포즈의 공분산 설정
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;                       // 중력, attitude의 공분산 설정
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;                        // 자이로스코프 bias, attitude 공분산 설정
  init_P(21,21) = init_P(22,22) = 0.00001;                                      // external parameter 공분산 설정
  kf_state.change_P(init_P);                                                    // esekfom.hpp P_에 init 공분산 행렬 전달
  last_imu_ = meas.imu.back();                                                  // 마지막 프레임의 데이터 저장해 UndistortPcl()에서 사용

}

// forward propagation, back propagation, de-distortion
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;                                // 현재 IMU 데이터
  v_imu.push_front(last_imu_);                          // 이전 프레임의 마지막 IMU를 현재 프레임 첫번째 IMU 앞에 추가
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec(); // 현재 프레임 처음 IMU time (= 이전 프레임 마지막 IMU 타임스탬프)
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();  // 현재 프레임 마지막 IMU time
  const double &pcl_beg_time = meas.lidar_beg_time;                 // pcl 시작 타임스탬프
  const double &pcl_end_time = meas.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  // 포인트 클라우드를 타임스탬프 기준으로 재정렬
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();   // 마지막 KF 추정의 posterior state 얻어서 IMU 예측의 초기 상태로 사용
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix())); //IMU pose에 초기 상태 추가

  /*** forward propagation at each imu point ***/
  // forward propagation에 해당하는 파라미터
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;  // time interval

  input_ikfom in;
  // 이 추정의 모든 IMU 측정값에 대해 확인하고 integration, discrete median method, forward propagation 수행
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);      // 이전 프레임의 IMU 데이터
    auto &&tail = *(it_imu + 1);  // 현재 프레임의 IMU 데이터
    // 현재 타임스탬프가 이전 프레임 라이다 종료 타임스탬프보다 빠르면 continue
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    // 현재 프레임과 다음 프레임의 평균
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    // 중력값으로 가속도 조정 : 초기화 단계에서 구한 mean_acc의 norm으로 normalization 한 후 중력 가속도 곱하므로 사용하는 IMU의 단위에 invariant
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    // IMU 시작 시간이 이전 프레임 마지막 라이다 시간보다 빠른 경우 (가장 앞에 마지막 IMU를 삽입하기 때문에 처음에 한 번 발생)
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      // 마지막 라이다 시간의 끝에서부터 propagation을 시작해 IMU와의 시간 차이를 계산
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      // 두 IMU 사이의 시간 차이
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    // 업데이트 된 오리지널 측정값의 중간값
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    // 공분산 행렬 구성
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    // IMU forward propagation, dt:각 propagation의 시간 간격
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    // IMU 예측 과정의 status
    imu_state = kf_state.get_x(); 
    angvel_last = angvel_avr - imu_state.bg;                // 계산한 각속도(중간값)와 예측한 각속도의 차이
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba); // 계산한 가속도(중간값)와 예측한 가속도의 차이를 IMU 좌표계로 변환
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];       // world 좌표계의 가속도를 얻기 위해 중력 추가
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;    // 마지막 IMU와 라이다 시작의 시간 간격
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix())); // IMU 예측값 저장
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // IMU 측정값의 마지막 프레임도 추가
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0; // 라이다 종료 시간이 IMU 보다 느리면 1, 빠르면 -1 
  dt = note * (pcl_end_time - imu_end_time);              // note를 곱하므로 dt는 항상 양수
  kf_state.predict(dt, Q, in);          // forward propagation에서 구한 state, 공분산에 대해 prediction.
  
  imu_state = kf_state.get_x();         // IMU state 업데이트
  last_imu_ = meas.imu.back();          // 다음 프레임에서 사용하기 위해 마지막 IMU 측정값 업데이트
  last_lidar_end_time_ = pcl_end_time;  // 다음 프레임에서 사용하기 위해 프레임의 마지막 라이다 측정값의 종료 시간 저장

  /*** undistort each lidar point (backward propagation) ***/
  // IMU 예측을 기반으로 라이다 포인트 클라우드 왜곡 제거
  if (pcl_out.points.begin() == pcl_out.points.end()) return; // 시작 포인트와 마지막 포인트가 같은 포인트이면 return
  auto it_pcl = pcl_out.points.end() - 1; // 포인트 클라우드의 가장 마지막 포인트를 가리키는 순환자(iterator)
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) // 예측한 모든 IMUpose에 대해 뒤에서부터 앞으로 backpropagation
  {
    auto head = it_kp - 1;    // 이전 프레임
    auto tail = it_kp;        // 현재 프레임
    R_imu<<MAT_FROM_ARRAY(head->rot);                             // 이전 프레임(head)의 IMU 회전 행렬
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl; 
    vel_imu<<VEC_FROM_ARRAY(head->vel);                           // 이전 프레임(head)의 IMU 속도
    pos_imu<<VEC_FROM_ARRAY(head->pos);                           // 이전 프레임(head)의 IMU position
    acc_imu<<VEC_FROM_ARRAY(tail->acc);                           // 현재 프레임(tail)의 IMU 가속도
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);                        // 현재 프레임(tail)의 IMU 각속도

    // 이전 IMU 프레임(head) 이후의 라이다 클라우드에 대해 왜곡 보정
    // 두 IMU 시간 사이에서 왜곡이 제거되기 때문에, 포인트 클라우드 시간이 이전 IMU 시간보다 느려야 한다.
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)    // sec 단위로 변환
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));       // 포인트의 회전
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);                                       // 포인트 position (라이다 좌표계)
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);     // 샘플링 시간과 스캔 종료 시간 간의 상대 position
                                                                                      //  = 계산한 IMU position(world) - 현재 IMU position(world)
      // 라이다 로컬 측정값을 라이다 scan-end 측정값으로 projection (T_ei는 T_i와 im_state_pose의 상대적인 translation이므로 회전 행렬 두 번 곱함)
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);
      // conjugate()는 회전 행렬의 켤레, rot.conjugate()는 쿼터니언의 켤레(회전의 inversion)
      // imu_state.offset_R_L_I : 라이다에서 IMU로의 회전 행렬 I^R_L
      // imu_state.offset_T_L_I : I^t_L

      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;  // 가장 처음 포인트까지 역전파했다면 break
    }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};       //현재 프레임의 IMU 측정값이 비어 있으면 return
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);      // first 라이다 프레임

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);    // IMU_init()에 기반해 scaling factor 곱함
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);   // forward propagation, back propagation, undistortion

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
