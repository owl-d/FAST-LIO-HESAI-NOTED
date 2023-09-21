// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <iostream>
#include <string>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};         // residual error, point-to-surface distance의 sum of squared
float DET_RANGE = 300.0f;               // 현재 라이다 시스템의 중심에서 각 맵 edge 까지의 거리 설정
const float MOV_THRESHOLD = 1.5f;       // 현재 라이다 시스템의 중심에서 각 맵 edge 까지의 가중치 설정
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;               // mutex lock
condition_variable sig_buffer;  // 조건값

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;  // 필터의 최소 크기, 맵의 최소 크기, viewing angle
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

// 포인트 클라우드 변수
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());             // 맵에서 feature points를 추출하고 ikd-tree에서 얻음
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());          // 왜곡 제거된 features
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());          // 왜곡 보정 후 다운샘플링 된 단일 프레임 포인트 클라우드 (라이다 시스템)
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());         // 왜곡 보정 후 다운샘플링 된 단일 프레임 포인트 클라우드 (world 시스템)
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));         // 맵의 포인트에 해당하는 feature poitn, local plane 파라미터 (world 시스템)
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));   // 왜곡 보정 후 다운샘플링 된 단일 프레임 포인트 클라우드 (body 시스템)
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));   // 대응하는 포인트 normal phasor
PointCloudXYZI::Ptr _featsArray;                                    // ikd-tree에서 맵에서 제거되어야 하는 포인트 클라우드 시퀀스

// 다운샘플링 된 복셀 포인트 클라우드
pcl::VoxelGrid<PointType> downSizeFilterSurf;   // 복셀 그리드를 이용한 단일 프레임 다운샘플링
pcl::VoxelGrid<PointType> downSizeFilterMap;    // 사용되지 않음

KD_TREE<PointType> ikdtree;                     // ikd-tree class

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);    // body 시스템의 x축 방향을 기준으로 한 라이다 포인트
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);   // world 시스템의 x축 방향을 기준으로 한 라이다 포인트
V3D euler_cur;                                  // 현재 오일러 각
V3D position_last(Zero3d);                      // 이전 프레임의 position
V3D Lidar_T_wrt_IMU(Zero3d);                    // T lidar to imu (imu = r * lidar + t)
M3D Lidar_R_wrt_IMU(Eye3d);                     // R lidar to imu (imu = r * lidar + t)

/*** EKF inputs and output ***/
// ESEKF 연산
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;    // state, noise dimension, input
state_ikfom state_point;                            // state
vect3 pos_lid;                                      // world 시스템에서 라이다 좌표

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;
//geometry_msgs::PoseStamped msg_body_pose_wrt_lidar;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();        // waiting queue에서 block 되어 있는 threads가 활성화된다.
}

// fast-lio2 information log
inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

// 포인트를 body에서 world 시스템으로 옮김 (ikfom의 position, attitude 이용)
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos); //라이다->IMU->world 좌표계로 변환

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// 포인트를 body에서 world 시스템으로 옮김
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// 포인트를 body에서 world 시스템으로 옮김
template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

// RGB를 포함한 포인트 클라우드를 body에서 world 시스템으로 옮김
void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// RGB를 포함한 포인트 클라우드를 body에서 IMU 시스템으로 옮김
void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

// eskf forward 결과를 얻고 맵이 너무 커져 메모리 오버플로우가 발생하지 않도록 동적으로 조절한다. (LOAM의 local map에서 feature 추출하는 것과 비슷)
BoxPointType LocalMap_Points;       // ikd-tree에서 local map의 바운딩 박스 코너 포인트
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();         // 삭제해야 하는 구역 클리어
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);    // XAix dividing point는 world 시스템으로 변환되어 사용 X
    V3D pos_LiD = pos_lid;                                  // 글로벌 시스템에서 라이다 위치
    // 로컬 맵 바운딩 박스의 코너 포인트 초기화 : world 시스템의 라이다 위치는 중심으로 하여 길이, 높이, 너비 각각 200*200*200
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // 라이다와 로컬 맵 바운더리 사이의 거리 (모든 방향, 또는 큐브 박스의 육면과의 거리)
    float dist_to_map_edge[3][2];
    bool need_move = false;
    // 현재 라이다 시스템의 중심에서 각 맵의 edge 까지의 거리
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 특정 방향까지에서 방향까지의 거리가 너무 작으면 논문의 Fig.3처럼 need_move=true
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return; // need_move가 false이면 움직이지 않고 return, need_move가 true이면 움직여야 하는 거리 계산
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;  // 새로운 로컬 맵 바운딩 박스 바운더리 포인트
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        // 바운딩 박스의 최소 바운더리 포인트로부터의 거리
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);    // 임시 바운딩 박스 제거
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    // 특정 박스 내의 포인트를 제거하기 위해 박스 사용
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

// 라이다 포인트 클라우드 콜백 함수 (AVIA 제외) : 데이터를 버퍼에 저장
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();  // lock
    scan_count ++;
    double preprocess_start_time = omp_get_wtime(); // time 저장
    if (msg->header.stamp.toSec() < last_timestamp_lidar)    // 현재 라이다 타임스탬프가 이전 라이다 타임스탬프보다 작으면
    {
        ROS_ERROR("lidar loop back, clear buffer");    // 에러 출력
        lidar_buffer.clear();                          // 라이다 버퍼 비우기
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);                           // 포인트 클라우드 전처리
    lidar_buffer.push_back(ptr);                        // 포인트 클라우드를 버퍼에 저장
    time_buffer.push_back(msg->header.stamp.toSec());   // time을 버퍼에 저장
    last_timestamp_lidar = msg->header.stamp.toSec();   // 마지막 time 저장
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time; // 전저리 시간
    mtx_buffer.unlock();
    sig_buffer.notify_all();    // 모든 스레드 활성화
}

double timediff_lidar_wrt_imu = 0.0;    // 라이다 time과 IMU time 차이
bool   timediff_set_flg = false;        // 시간 동기화 flag (false : 시간 동기화 수행 안 되어 있음, true : 시간 동기화가 수행 되었음)
// sub_pcl의 콜백 함수 : Livox 데이터를 받아 전처리(feature 추출, 다운샘플링, 필터링)하고 라이다 데이터 queue에 저장
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();  // lock
    double preprocess_start_time = omp_get_wtime(); // 포인트 클라우드 전처리 시작 시간
    scan_count ++;  // 라이다 스캔 총 카운트
    // 현재 스캔의 타임스탬프가 이전 스캔보다 빠르면 라이다 데이터 캐시 queue를 비운다.
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    // 시간 동기화 플래그가 false이고 IMU와 라이다의 타임스탬프가 10s 이상 차이나면 에러 메세지를 띄운다.
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    // 시간 동기화 플래그가 true이고 IMU와 라이다의 타임스탬프가 1s 이상 차이나면
    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;    // 시간 동기화 여부를 알려주는 flag를 true로 설정
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;    // 두 타임스탬프의 차이 계산(이때 0.1을 더함) -> imu_cbk에서 사용
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI()); // 수신한 라이다 데이터를 pcl 포인트 클라우드 형식에 저장
    p_pre->process(msg, ptr);   // p_pre : 전처리 클래스의 smart pointer
    lidar_buffer.push_back(ptr);                    // ptr(=pl_surf) 저장
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time; // 포인트 클라우드 전처리에 걸린 총 시간
    mtx_buffer.unlock();
    sig_buffer.notify_all();    // 모든 스레드 활성화
}

// subscriber sub_imu의 콜백함수 : IMU 데이터를 수신해 IMU 데이터 캐시 queue에 저장
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    // IMU와 라이다 타임스탬프 조정
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) // 시간 동기화 flag가 true이고 라이다와 IMU 사이의 타임스탬프 차이가 0.1 이상이면
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec()); // IMU 타임스탬프를 라이다 타임스탬프에 맞춤
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    // 현재 IMU 타임스탬프가 이전 IMU보다 이전이면 IMU 데이터가 잘못된 것이므로 imu_buffer 비움
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();        // unlock
    sig_buffer.notify_all();
}

void save_path(const nav_msgs::Path::ConstPtr &msg)
{   
    static int cnt = 0;
    string line;
    auto time = msg->poses[cnt].header.stamp.toSec();
    auto x = msg->poses[cnt].pose.position.x;
    auto y = msg->poses[cnt].pose.position.y;
    auto z = msg->poses[cnt].pose.position.z;
    auto qx = msg->poses[cnt].pose.orientation.x;
    auto qy = msg->poses[cnt].pose.orientation.y;
    auto qz = msg->poses[cnt].pose.orientation.z;
    auto qw = msg->poses[cnt].pose.orientation.w;
    line = to_string(time)+" "+to_string(x)+" "+to_string(y)+" "+to_string(z)+" "+to_string(qx)+" "+to_string(qy)+" "+to_string(qz)+" "+to_string(qw)+" ";
    //cout << cnt+1 << " " << line << endl;
    
    FILE *fp;
    string dir = root_dir + "/trajectories/exp.txt";
    if (cnt == 0)
    {
        fp = fopen(dir.c_str(), "w");
    }
    else
    {
        fp = fopen(dir.c_str(), "a");
    }
    fprintf(fp, (line+"\n").c_str());
    fclose(fp);
    
    cnt++; 
}

// 버퍼에 있는 데이터를 처리하고, 두 라이다 사이의 IMU 데이터를 꺼내 조정하고 meas에 저장한다. 
double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {   // lidar_buffer 또는 imu_buffer 데이터가 없으면, return false
        return false;
    }

    /*** push a lidar scan ***/
    // meas에 넣을 라이다 데이터가 없으면 아래의 연산을 수행한다.
    if(!lidar_pushed)    // lidar_pushed=false가 디폴트
    {
        meas.lidar = lidar_buffer.front();            // lidar_buffer 포인트 클라우드를 meas에 저장 
        meas.lidar_beg_time = time_buffer.front();    // lidar 데이터의 begin/end time meas에 저장
        if (meas.lidar->points.size() <= 1) // 라이다에 포인트 클라우드가 없으면 return false
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);    // 라이다 end time = 라이다 타임스탬프 + (라이다 데이터 안 마지막 포인트의 time)
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time; // 라이다 측정 시작 타임스탬프

        lidar_pushed = true;    // 라이다 측정을 성공적응로 추출했다는 flag
    }

    // 마지막 IMU 타임스탬프(queue의 마지막)는 마지막 라이다 타임스탬프보다 빠를 수 없다. (타임 스탬프 차이를 계산할 때 last_timestamp_imu에 0.1을 더하기 때문)
    // IMU의 Hz가 라이다보다 더 크고, 라이다를 IMU 기준으로 맞추기 때문에(라이다 end time 뒤에 있는 IMU로 투영) IMU가 더 최신이어야 한다.
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    // lidar_beg_time 과 lidar_end_time 사이의 모든 IMU 데이터에 대해
    // IMU 타임스탬프가 라이다 종료 타임스탬프보다 빠르면, 이번 프레임의 IMU 데이터를 표현하여 meas에 저장된다.
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();    // IMU 데이터의 타임스탬프
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());     // IMU 데이터를 meas에 넣음
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();   // 라이다 데이터 pop
    time_buffer.pop_front();    // 타임스탬프 pop
    lidar_pushed = false;       // 라이다 데이터가 meas에 위치해있다.
    return true;
}

int process_increments = 0;
void map_incremental()  // 맵의 incremental update. 주로 ikd-tree의 맵을 생성한다.
{
    PointVector PointToAdd;             // ikd-tree에 추가되어야 하는 포인트 클라우드
    PointVector PointNoNeedDownsample;  // ikd-tree를 추가할 때, 포인트 클라우드를 다운샘플링 할 필요가 없다.
    PointToAdd.reserve(feats_down_size);    // 구성된 맵 포인트
    PointNoNeedDownsample.reserve(feats_down_size); // 구성된 맵 포인트, 다운샘플링이 필요 없는 포인트 클라우드

    // 포인트와 바운딩 박스의 중심점 사이의 거리에 기반해, 다운샘플링 필요한지 여부
    for (int i = 0; i < feats_down_size; i++)
    {
        // world 좌표계로 변환
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        // 맵에 추가해야 하는 키 포인트가 있는지 결정
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i]; // 가까운 포인트 얻음
            bool need_add = true;                               // 맵에 추가해야 하는지 여부
            BoxPointType Box_of_Point;                          // 포인트가 위치한 박운딩 박스
            PointType downsample_result, mid_point;             // 다운샘플링 결과, midpoint(feature point가 속한 그리드의 중심점 좌표)
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point); // 현재 포인트와 박스 중심 사이의 거리

            // x,y,z 세 방향에서 가장 가까운 포인트와 중심 사이의 거리를 계산하고, 합칠 때 다운샘플링 필요한지 결정
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);   // 세 방향에서 거리가 맵 그리드의 semi-axis 길이보다 크면 다운샘플링 필요 X
                continue;
            }
            // NUM_MATCH_POINTS 개의 주변 포인트와 바운딩 박스 중심점 사이의 range 계산
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;       //인접 포인트 수가 NUM_MATCH_POINTS 보다 적으면 break하고 바로 PointToAdd 수행.
                if (calc_dist(points_near[readd_i], mid_point) < dist)  //이웃 포인트와 중심까지의 거리가 현재 포인트와 중심까지의 거리보다 가까우면, 현재 포인트 추가할 필요 X
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);    // PointToAdd에 추가
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);  // 주변에 포인트가 없거나 EKF가 초기화 되지 않았으면, PointToAdd에 추가
        }
    }

    double st_time = omp_get_wtime();                                   // 시작 시간 기록
    add_point_size = ikdtree.Add_Points(PointToAdd, true);              // 포인트 합칠 때 다운샘플링 필요
    ikdtree.Add_Points(PointNoNeedDownsample, false);                   // 포인트 합칠 때 다운샘플링 불필요
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();  // ikd-tree에 추가된 총 포인트 수 계산
    kdtree_incremental_time = omp_get_wtime() - st_time;                // kd-tree 생성 시간 업데이트
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));        // publish 되기를 대기하는 포인트 클라우드
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());                // 저장되기를 대기하는 포인트 클라우드
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en) // 라이다 데이터 publish 여부, dense 데이터 publish 여부, 라이다 데이터의 body 데이터 publish 여부
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);    // 다운샘플링 필요한지 판단
        int size = laserCloudFullRes->points.size();                                                // 변환할 포인트 클라우드 사이즈
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));   // world 좌표계로 변환한 포인트 클라우드

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);       // world 좌표계로 변환
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

// 왜곡 보정 라이다 시스템을 IMU 시스템으로 변환
void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1)); // IMU 시스템으로 변환한 포인트 클라우드

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);     // IMU 좌표계로 변환
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

// active featrue point를 맵에 전달
void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));     // h_share_model에서 얻은 world 시스템으로 변환한 포인트 클라우드
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

// ikd-tree 맵 publish
void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

// output t,q를 설정. publish_odometry와 publish_path에서 호출
template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);   // eskf에서 얻은 position
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;         // eskf에서 얻은 quaternion
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

// odometry publish
void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        // 공분산 행렬 P : 회전 다음 위치 순서 / POSE는 위치 다음 회전 순서이므로 공분산 반전시켜야 함.
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );   //tf 변환 publish
}

// path publish
void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    //static int jjj = 0;
    //if (jjj % 10 == 0) 
    //{
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
    //}
    //jjj++;
}

// residual 정보 계산
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();   // match 시작 타임
    laserCloudOri->clear();                 // body 시스템의 유효 포인트 클라우드 비우기
    corr_normvect->clear();                 // 대응하는 법선 벡터 비우기
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    // 다운샘플링 후 각 feature point에 대해 residual 계산
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];    // 다운샘플링 후의 각 feature point
        PointType &point_world = feats_down_world->points[i];   // 다운샘플링 후의 각 feature point의 world 좌표계

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos); // residual 계산하기 위해 world 좌표계로 변환
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)    // 수렴했다면
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            // NN(nearest neighbor) 포인트 수가 NUM_MATCH_POINTS 이하이거나 NN과 feature point 사이의 거리가 5m 보다 크면 false (= 평면이 아니다)
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;  // false이면 continue, true이면(=평면일 가능성이 있다) 아래 과정에서 평면인지 판단

        VF(4) pabcd;                                // plane 포인트 정보
        point_selected_surf[i] = false;             // 일단 false로 해두고 평면 포인트 맞다고 판단되면 나중에 true로 변경함

        // 평면 방정식 ax+by+cz+d=0을 피팅하고 point-to-plane 거리 계산
        if (esti_plane(pabcd, points_near, 0.1f))   // 평면 포인트 법선 벡터 찾기 (common_lib.h)
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);  // point-to-plane 거리 계산
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());                                                    // 0-1 사이의 값, range가 클수록 평면에서 좀 더 멀어도 허용해줌

            if (s > 0.9)    // residual이 threshold보다 크면 평면 위의 포인트가 맞다고 판단. s가 1에 가까울수록 평면 위에 있다는 의미이다.
            {
                point_selected_surf[i] = true;      // valid point flag
                normvec->points[i].x = pabcd(0);    // 법선 벡터 저장 (residual 계산에 필요)
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2; // 법선 벡터의 intensity에 point-to-plane 거리 저장
                res_last[i] = abs(pd2);             // residual 저장
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i]) // 평면 위의 포인트이면
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; // 바디 프레임 기준 surf feature point
            corr_normvect->points[effct_feat_num] = normvec->points[i];         // 평면의 법선 벡터
            total_residual += res_last[i];          // 총 residual 계산
            effct_feat_num ++;                      // effective feature points 카운트 + 1
        }
    }

    if (effct_feat_num < 1)                 // surf feature point 없으면 return
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;    // residual의 평균
    match_time  += omp_get_wtime() - match_start;       // match 시작 후 현재까지 걸린 시간 반환
    double solve_start_  = omp_get_wtime();             // solving 시작 시간
    
    /*** Computation of Measuremnt Jacobian matrix H=J*P*J' and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);    // measurement model h의 자코비안 H. 논문에서 eq. (23)
    ekfom_data.h.resize(effct_feat_num);                    // measurement vector h

    // 관측값과 에러의 자코비안 행렬 구하기. eq. (14, 12, 13)
    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];               // valid 포인트 좌표
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;                                              // 포인트의 antisymmetric 행렬 계산
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);                 // point value를 cross product 행렬로 변환
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;   // IMU 좌표계로 변환
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);                         // IMU midpoint의 antisymmetric 행렬 계산

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        // 자코비안 행렬 J 계산은 fast-lio v1의 eq (14) 
        // Fast-lio2 : ieskf에서 상태 업데이트는 residual function의 최적화 문제로 볼 수 있다.
        // LINS에서는 LOAM의 가우스-뉴턴 방법을 사용했는데 fast-lio2는 이를 ieskf의 반복 업데이트 프로세스로 대체했다.
        V3D C(s.rot.conjugate() *norm_vec);     // R^(-1)*법선 벡터
        V3D A(point_crossmat * C);              // IMU 좌표계 포인트 C * antisymmetric point
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);   // 변수명에 be가 있으면 라이다 좌표계, 없으면 IMU 좌표계
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;    // point-to-plane 거리
    }
    solve_time += omp_get_wtime() - solve_start_;   // solve 에 걸린 시간
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    // 사용 X
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());    // 사용 X

    memset(point_selected_surf, true, sizeof(point_selected_surf)); // point_selected_surf 배열(plane point 선택에 사용)의 모든 요소를 true로 설정
    memset(res_last, -1000.0f, sizeof(res_last));                   // res_last 배열(plane fitting에 사용)의 모든 요소를 -1000.0로 설정
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);   // 복셀그리드 필터 파라미터 설정 (최소 복셀 사이즈로 설정)
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);       // 복셀그리드 필터 파라미터 설정
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);                       // 라이다에서 IMU로의 extrinsic 파라미터
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);         // IMU 파라미터 설정, p_imu(ImuProcess의 스마트 포인터) 초기화
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);         // epsi(수렴 threshold)의 모든 배열 요소 값을 0.001로 설정
    // h_dyn_share_in()으로 측정값(z), 예측 측정값(h), 편미분 행렬(h_x, h_v), 노이즈 공분산(R) 계산
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);    //함수 주소를 kf 객체에 전달.

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);                          // 라이다 포인트 클라우드 sub
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);             // IMU 토픽 sub

    ros::Subscriber sub_path = nh.subscribe("/path",100000, save_path);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>       // 현재 스캔한 포인트 클라우드
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>  // IMU 좌표계로 모션 왜곡이 보정된 포인트 클라우드
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);  // 인터럽트 처리 함수. 인터럽트가 발생한 경우 SigHandle() 호출.
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)  // main loop
    {
        if (flg_exit) break;        // 인터럽트가 발생하면 main loop 종료
        ros::spinOnce();
        if(sync_packages(Measures))     // buffer에 있는 라이다, IMU 데이터를 삭제하고 시간 정렬을 수행한 후 Measures에 저장
        {
            if (flg_first_scan)     // 첫 라이다 스캔이면
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);  // IMU 데이터 전처리 : 포인트 클라우드 왜곡 처리, forward propagation, back propagation
            state_point = kf.get_x();                       // kf 예측의 global state (IMU)
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;     // world 좌표계에서 라이다 시스템의 position. 
                                                                                        // W^p_L = W^p_I + W^R_I * I^t_L

            if (feats_undistort->empty() || (feats_undistort == NULL))  // 포인트 클라우드가 비어있으면 라이다가 아직 왜곡 보정을 완료하지 않은 것이고 현재는 완전한 초기화가 불가능하다.
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            // 초기화가 완료되었는지 확인 : 첫 스캔 시간과 첫 포인트 클라우드 시간 차이가 INIT_TIME보다 커야 한다.
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            // eskf feedforward 결과를 얻은 후 로컬 맵을 동적으로 조정
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);  // 왜곡 보정된 포인트 클라우드
            downSizeFilterSurf.filter(*feats_down_body);        // 다운샘플링으로 필터링 된 포인트 클라우드
            t1 = omp_get_wtime();                               // 시간 기록
            feats_down_size = feats_down_body->points.size();   // 필터링 된 포인트 수
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);  // ikd-tree의 다운샘플링 파라미터 설정
                    feats_down_world->resize(feats_down_size);          // 다운샘플링으로 얻은 맵 포인트 수가 body 시스템과 일치하도록 유지한다.
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));    // 다운샘플링 된 맵 포인트를 world 좌표계로 변환
                    }
                    ikdtree.Build(feats_down_world->points);    // ikd-tree build
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();   // ikd-tree의 유효 노드 수. invalid 포인트는 삭제되었다고 표시되어 있다.
            kdtree_size_st = ikdtree.size();            // ikd-tree의 노드 수
            
            //cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            // external 파라미터, 회전 행렬 -> 오일러 각
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;   // 예측 결과 출력

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);   // PCL_Storage의 메모리 release
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);    // 시각화를 위해 tree를 flatten
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);    // 인덱스 탐색
            Nearest_Points.resize(feats_down_size);         // 다운샘플링 된 포인트를 사용해 closest point 탐색
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);   // Iterative KF 업데이트, 맵 정보 업데이트
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;                             // 전체 프로세스에 걸린 시간
                s_plot2[time_log_counter] = feats_undistort->points.size();     // feature points 수
                s_plot3[time_log_counter] = kdtree_incremental_time;            // kdtree incremental 시간
                s_plot4[time_log_counter] = kdtree_search_time;                 // kdtree 탐색 시간
                s_plot5[time_log_counter] = kdtree_delete_counter;              // kdtree에서 삭제된 포인트 수
                s_plot6[time_log_counter] = kdtree_delete_time;                 // kdtree 삭제 시간
                s_plot7[time_log_counter] = kdtree_size_st;                     // kdtree 초기 사이즈
                s_plot8[time_log_counter] = kdtree_size_end;                    // kdtree 마지막 사이즈
                s_plot9[time_log_counter] = aver_time_consu;                    // 걸린 평균 시간
                s_plot10[time_log_counter] = add_point_size;                    // add point 수
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
