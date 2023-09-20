#include "preprocess.h"

#define RETURN0     0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess()
  :feature_enabled(0), lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
  inf_bound = 10;  // 유효한 포인트 집합을 판단하는 threshold. inf_bound 이상이면 블라인드 영역.
  N_SCANS   = 6;   // 라이다의 라인 수
  SCAN_RATE = 10;  // 스캔 rate (10Hz)
  group_size = 8;  // 한 그룹에 8개의 포인트
  disA = 0.01;     // 한 포인트 집합이 plane 인지 판단하는 threshold
  disB = 0.1;      // 한 포인트 집합이 plane 인지 판단하는 threshold
  p2l_ratio = 225; // 포인트에서 라인까지의 거리 threshold. p2l_ratio 보다 작아야 표면을 구성한다고 판단.
  limit_maxmid =6.25; //midpoint에서 왼쪽까지의 거리 변화율 범위
  limit_midmin =6.25; //midpoint에서 오른쪽까지의 거리 변화율 범위
  limit_maxmin = 3.24; //왼쪽에서 오른쪽까지의 거리 변화율 범위
  jump_up_limit = 170.0;
  jump_down_limit = 8.0;
  cos160 = 160.0;
  edgea = 2;    //포인트 간의 거리가 거리의 2m를 넘으면 가려진 것으로 판단
  edgeb = 0.1;  // 포인트 간의 거리가 0.1m를 넘으면 block 된 것으로 판단
  smallp_intersect = 172.5;
  smallp_ratio = 1.2;     // 세 포인트의 각도가 172.5 deg 보다 크고 비율이 1.2보다 작으면 평면으로 판단
  given_offset_time = false;  // time offset 존재 여부

  jump_up_limit = cos(jump_up_limit/180*M_PI);       // 각도가 170 deg 이상의 포인트들은 스킵하고 포함된 것으로 간주한다.
  jump_down_limit = cos(jump_down_limit/180*M_PI);   // 각도가 8 deg 이하의 포인트들은 스킵한다.
  cos160 = cos(cos160/180*M_PI);                     // 각도 limit
  smallp_intersect = cos(smallp_intersect/180*M_PI); // 세 포인트의 각도가 172.5 deg 이상고 비율이 1.2보다 작으면 평면으로 판단
}

Preprocess::~Preprocess() {}

void Preprocess::set(bool feat_en, int lid_type, double bld, int pfilt_num)
{
  feature_enabled = feat_en;  // frature point를 추출 여부
  lidar_type = lid_type;      // 라이다 종류
  blind = bld;                // 최소 거리 threshold. 0~blind 범위의 포인트 클라우드 필터링
  point_filter_num = pfilt_num; // 샘플링 주기. point_filter_num 개의 포인트마다 한 포인트를 얻는다.
}

// Livox 라이다 포인트 클라우드 전처리 함수
void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{  
  avia_handler(msg);
  *pcl_out = pl_surf;
}

// 시간 단위, 라이다 종류 설정
void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  switch (time_unit)
  {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  switch (lidar_type)
  {
  case OUST64:
    oust64_handler(msg);
    break;

  case VELO16:
    velodyne_handler(msg);
    break;

  case HESAI:
    hesai_handler(msg);
    break;

  default:
    printf("Error LiDAR Type");
    break;
  }
  *pcl_out = pl_surf;
}

// 라이다 포인트 클라우드 데이터 전처리
void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
  // 이전 포인트 클라우드 캐시 제거
  pl_surf.clear();  // 이전 planar 포인트 클라우드 캐시 제거
  pl_corn.clear();  // 이전 corner 포인트 클라우드 캐시 제거
  pl_full.clear();  // 이전 전체 포인트 클라우드 캐시 제거
  double t1 = omp_get_wtime(); // 이후에 사용하지 않음
  int plsize = msg->point_num; // 프레임의 전체 포인트 수
  // cout<<"plsie: "<<plsize<<endl;

  pl_corn.reserve(plsize); // 공간 할당
  pl_surf.reserve(plsize); // 공간 할당
  pl_full.resize(plsize);  // 공간 할당

  for(int i=0; i<N_SCANS; i++)
  {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);  // 각 스캔에 저장된 포인트 수
  }
  uint valid_num = 0;    // 유효 포인트 수
  
  // feature extraction : 기본적으로 FAST-LIO2는 feature 추출하지 않음
  if (feature_enabled)
  {
    for(uint i=1; i<plsize; i++)  // 각 포인트에 대해 처리
    {
      // 0 ~ N_SCANS 사이의 라인에 있고, echo 순서가 0또는 1인 포인트만 처리
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        pl_full[i].x = msg->points[i].x; // 포인트의 x축 좌표
        pl_full[i].y = msg->points[i].y; // 포인트의 y축 좌표
        pl_full[i].z = msg->points[i].z; // 포인트의 z축 좌표
        pl_full[i].intensity = msg->points[i].reflectivity; // 포인트의 intensity
        pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // 곡률에 각 레이저 포인트의 시간을 저장하여 사용한다.

        bool is_new = false;
        // 현재 포인트와 이전 포인트 사이의 거리가 충분히 크면(> 1e-7) 현재 포인트가 useful 하다고 판단해 대응하는 라인의 pl_buff queue에 추가한다.
        if((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
            || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
            || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
        {
          pl_buff[msg->points[i].line].push_back(pl_full[i]);  // 대응하는 라인의 pl_buff queue에 현재 포인트를 추가
        }
      }
    }
    static int count = 0;
    static double time = 0.0;
    count ++;
    double t0 = omp_get_wtime();
    // 라이다의 각 라인에 대해 처리
    for(int j=0; j<N_SCANS; j++)
    {
      // 라인에 속한 포인트 수가 너무 작으면 (<=5) 다음 라인으로 continue
      if(pl_buff[j].size() <= 5) continue;
      pcl::PointCloud<PointType> &pl = pl_buff[j];
      plsize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(plsize);
      plsize--;
      for(uint i=0; i<plsize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y); // 각 포인트에서 로봇까지의 거리 계산
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = sqrt(vx * vx + vy * vy + vz * vz); // 두 포인트 pl[i]와 pl[i+1] 사이의 거리 계산
      }
      // 마지막 포인트 i는 i_1을 가지고 있지 않으므로 range 계산 (두 포인트 간의 거리는 계산하지 않음)
      types[plsize].range = sqrt(pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y);
      give_feature(pl, types);
      // pl_surf += pl;
    }
    time += omp_get_wtime() - t0;
    printf("Feature extraction time: %lf \n", time / count);
  }
  else
  {
    // 각 포인트를 개별적으로 처리
    for(uint i=1; i<plsize; i++)
    {
      // 0 ~ N_SCANS 사이의 라인에 있고, echo 순서가 0또는 1인 포인트만 처리
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        valid_num ++; // 유효 포인트 수 + 1
        if (valid_num % point_filter_num == 0) // point_filter_num(e.g. avia : 3)마다 한 포인트씩 (= 1/3 points filtering)
        {
          pl_full[i].x = msg->points[i].x; // 포인트의 x 좌표
          pl_full[i].y = msg->points[i].y; // 포인트의 y 좌표
          pl_full[i].z = msg->points[i].z; // 포인트의 z 좌표
          pl_full[i].intensity = msg->points[i].reflectivity; // 포인트의 intensity
          pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // 곡률에 각 레이저 포인트의 시간 저장해 사용, curvature unit: ms

          // 현재 포인트와 이전 포인트 사이의 거리가 크고(> 1e-7)(= 둘은 다른 포인트) 포인트까지의 거리가 블라인드 영역 밖이면 현재 포인트가 useful 하다고 판단
          if(((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
              || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
              || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
              && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind)))
          {
            pl_surf.push_back(pl_full[i]);  // pl_surf 에 추가
          }
        }
      }
    }
  }
}

void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  if (feature_enabled)
  {
    for (int i = 0; i < N_SCANS; i++)
    {
      pl_buff[i].clear();
      pl_buff[i].reserve(plsize);
    }

    for (uint i = 0; i < plsize; i++)
    {
      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      if (range < (blind * blind)) continue;
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.3;
      // atan2()는 -pi ~ pi 범위를 반환하므로 쓸모 X
      if (yaw_angle >= 180.0)
        yaw_angle -= 360.0;
      if (yaw_angle <= -180.0)
        yaw_angle += 360.0;

      added_pt.curvature = pl_orig.points[i].t * time_unit_scale;
      if(pl_orig.points[i].ring < N_SCANS)
      {
        pl_buff[pl_orig.points[i].ring].push_back(added_pt);
      }
    }

    for (int j = 0; j < N_SCANS; j++)
    {
      PointCloudXYZI &pl = pl_buff[j];
      int linesize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = vx * vx + vy * vy + vz * vz;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
      give_feature(pl, types);
    }
  }
  else
  {
    double time_stamp = msg->header.stamp.toSec();
    // cout << "===================================" << endl;
    // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
      if (i % point_filter_num != 0) continue;

      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      
      if (range < (blind * blind)) continue;
      
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0; // 주어진 포인트가 위치한 샘플 표면의 샘플 표면의 법선 방향과 해당 곡률 측정값
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.curvature = pl_orig.points[i].t * time_unit_scale; // curvature unit: ms

      pl_surf.points.push_back(added_pt);
    }
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE;       // scan angular velocity
    std::vector<bool> is_first(N_SCANS,true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);      // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);     // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);    // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0) // 각 포인트의 타임스탬프가 제공된다면
    {
      given_offset_time = true;
    }
    else // 각 포인트에 대한 타임스탬프가 제공되지 않는다면
    {
      given_offset_time = false;
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end  = yaw_first;
      int layer_first = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578; // 종료 yaw 계산
          break;
        }
      }
    }

    if(feature_enabled)
    {
      for (int i = 0; i < N_SCANS; i++)
      {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
      }
      
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        int layer  = pl_orig.points[i].ring;
        if (layer >= N_SCANS) continue;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale; // units: ms

        if (!given_offset_time) // 타임스탬프가 주어져있지 않다면
        {
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
          if (is_first[layer]) // 각 라인에 first yaw를 지정하고 시간을 0으로 되돌린다.
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;            // 각 라인의 마지막 포인트 시간 지정
              time_last[layer]=added_pt.curvature;  // 각 라인의 마지막 포인트 시간 지정
              continue;
          }

          // 각 포인트의 시간 계산
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          // 새롭게 얻은 포인트의 시간은 직전의 포인트보다 빠를 수 없다.
          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        pl_buff[layer].points.push_back(added_pt);
      }

      for (int j = 0; j < N_SCANS; j++)
      {
        PointCloudXYZI &pl = pl_buff[j];
        int linesize = pl.size();
        if (linesize < 2) continue;  //스킵할 포인트가 매우 적음
        vector<orgtype> &types = typess[j];
        types.clear();
        types.resize(linesize);
        linesize--;
        for (uint i = 0; i < linesize; i++)
        {
          types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
          vx = pl[i].x - pl[i + 1].x;
          vy = pl[i].y - pl[i + 1].y;
          vz = pl[i].z - pl[i + 1].z;
          types[i].dista = vx * vx + vy * vy + vz * vz;
        }
        types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
        give_feature(pl, types);
      }
    }
    else
    {
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
        
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

        if (!given_offset_time)
        {
          int layer = pl_orig.points[i].ring;
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          // compute offset time
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
          if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
          {
            pl_surf.points.push_back(added_pt);
          }
        }
      }
    }
}

void Preprocess::hesai_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<hesai_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);
    
    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE;       // scan angular velocity
    std::vector<bool> is_first(N_SCANS,true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);      // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].timestamp > 0)
    {
      given_offset_time = true;
    }
    else
    {
      given_offset_time = false;
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end  = yaw_first;
      int layer_first = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }

    double time_head = pl_orig.points[0].timestamp;
    
    for (int i = 0; i < plsize; i++)
    {
      PointType added_pt;
      // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
      
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature = (pl_orig.points[i].timestamp - time_head) * 1000.f; // time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;
      if (!given_offset_time)
      {
        int layer = pl_orig.points[i].ring;
        double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

        if (is_first[layer])
        {
          // printf("layer: %d; is first: %d", layer, is_first[layer]);
            yaw_fp[layer]=yaw_angle;
            is_first[layer]=false;
            added_pt.curvature = 0.0;
            yaw_last[layer]=yaw_angle;
            time_last[layer]=added_pt.curvature;
            continue;
        }

        // compute offset time
        if (yaw_angle <= yaw_fp[layer])
        {
          added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
        }
        else
        {
          added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
        }

        if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

        yaw_last[layer] = yaw_angle;
        time_last[layer]=added_pt.curvature;
      }

      if (i % point_filter_num == 0)
      {
        if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
        {
          pl_surf.points.push_back(added_pt);
        }
      }
    }
    
}

// 각 라인의 포인트 클라우드에서 feature 추출
// pl : pcl format의 포인트 클라우드. 스캔 라인 위의 포인트를 입력.
// types : 포인트 클라우드의 다른 속성
void Preprocess::give_feature(pcl::PointCloud<PointType> &pl, vector<orgtype> &types)
{
  int plsize = pl.size(); // 단일 라인의 포인트 수
  int plsize2;
  if(plsize == 0)
  {
    printf("something wrong\n");
    return;
  }
  uint head = 0;

  // 블라인드 영역 안에 있을 수 없으며, 이 라인의 블라인드 영역이 아닌 포인트부터 시작한다.
  while(types[head].range < blind)
  {
    head++;
  }

  // Surf
  plsize2 = (plsize > group_size) ? (plsize - group_size) : 0; // 현재 포인트 뒤에 8개의 포인트가 있는지 확인한다. 충분하다면 점점 줄인다.

  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero()); // 현재 plane의 법선 벡터
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero()); // 이전 plane의 법선 벡터

  uint i_nex = 0, i2;   // i2는 현재 포인트의 다음 포인트
  uint last_i = 0;      // last_i는 이전 포인트의 저장된 인덱스
  uint last_i_nex = 0;  // last_i_nex는 이전 포인트 후 다음 포인트의 인덱스
  int last_state = 0;   // 1이면 이전 state가 plane 이라는 뜻이고, 아니면 0
  int plane_type;

  for(uint i=head; i<plsize2; i++) // plane 인지 확인하기 위해 8개 포인트 가져오기
  {
    if(types[i].range < blind) // 블라인드 영역 내의 포인트는 처리하지 않는다.
    {
      continue;
    }

    i2 = i; //i2 업데이트

    plane_type = plane_judge(pl, types, i, i_nex, curr_direct); // plane을 찾고 타입(0,1,2)를 반환한다.
    
    if(plane_type == 1) // Return 1. 일반적으로 plane이 기본값이다.
    {
      // determined plane 포인트, possible plane 포인트 설정
      for(uint j=i; j<=i_nex; j++)
      { 
        if(j!=i && j!=i_nex)
        {
          types[j].ftype = Real_Plane; // determined points = 시작 포인트와 끝 보인트 사이의 모든 포인트
        }
        else
        {
          types[j].ftype = Poss_Plane; // possible points = 시작 포인트와 끝 포인트
        }
      }
      
      // if(last_state==1 && fabs(last_direct.sum())>0.5)

      // 처음에는 last_state=0을 직접 스킵
      // 그 뒤에는 last_state=1
      // 이전 state가 평면이었으면, 현재 포인트가 두 평면의 edge에 있는지 상대적으로 평평한 평면에 있는지 확인
      if(last_state==1 && last_direct.norm()>0.1)
      {
        double mod = last_direct.transpose() * curr_direct;
        if(mod>-0.707 && mod<0.707)
        {
          // 두 표면의 법선 벡터 사이의 각도가 45 ~ 135 deg 가 되도록 ftype 수정함으로써 두 평면의 edge에 있는 포인트로 간주
          types[i].ftype = Edge_Plane;
        }
        else
        {
          // 아니면 real 평면 포인트로 간주
          types[i].ftype = Real_Plane;
        }
      }
      
      i = i_nex - 1;
      last_state = 1;
    }
    else // plane_type이 0 또는 2일 때
    {
      i = i_nex;
      last_state = 0; // 평면 포인트가 아닌 것으로 설정
    }
    // else if(plane_type == 0)
    // {
    //   if(last_state == 1)
    //   {
    //     uint i_nex_tem;
    //     uint j;
    //     for(j=last_i+1; j<=last_i_nex; j++)
    //     {
    //       uint i_nex_tem2 = i_nex_tem;
    //       Eigen::Vector3d curr_direct2;

    //       uint ttem = plane_judge(pl, types, j, i_nex_tem, curr_direct2);

    //       if(ttem != 1)
    //       {
    //         i_nex_tem = i_nex_tem2;
    //         break;
    //       }
    //       curr_direct = curr_direct2;
    //     }

    //     if(j == last_i+1)
    //     {
    //       last_state = 0;
    //     }
    //     else
    //     {
    //       for(uint k=last_i_nex; k<=i_nex_tem; k++)
    //       {
    //         if(k != i_nex_tem)
    //         {
    //           types[k].ftype = Real_Plane;
    //         }
    //         else
    //         {
    //           types[k].ftype = Poss_Plane;
    //         }
    //       }
    //       i = i_nex_tem-1;
    //       i_nex = i_nex_tem;
    //       i2 = j-1;
    //       last_state = 1;
    //     }

    //   }
    // }

    last_i = i2;                // last_i 업데이트
    last_i_nex = i_nex;         // last_i_nex 업데이트
    last_direct = curr_direct;  // last_direct 업데이트
  }

  // edge 포인트 찾기
  plsize2 = plsize > 3 ? plsize - 3 : 0;  // 남아있는 포인트가 3개 이하이면 edge 포인트를 찾지 않는다. 아니면 edge 포인트인지 계산
  for(uint i=head+3; i<plsize2; i++)
  {
    if(types[i].range<blind || types[i].ftype>=Real_Plane) // 포인트는 블라인드 영역에 있을 수 없고 normal 포인트, possible plane 포인트에 속해야 한다.
    {
      continue;
    }

    if(types[i-1].dista<1e-16 || types[i].dista<1e-16) // 이 포인트와 이전 포인트 사이의 거리는 너무 가까울 수 없다.
    {
      continue;
    }

    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z); // 현재 포인트로 구성된 벡터
    Eigen::Vector3d vecs[2];

    for(int j=0; j<2; j++)
    {
      int m = -1;
      if(j == 1)
      {
        m = 1;
      }

      if(types[i+m].range < blind) // 이전/다음 포인트가 블라인드 영역(4m) 이라면
      {
        if(types[i].range > inf_bound) // 10m 이상이면
        {
          types[i].edj[j] = Nr_inf; // Nr_inf 포인트 지정 (더 멀리 점프)
        }
        else
        {
          types[i].edj[j] = Nr_blind; // Nr_blind 포인트 지정 (블라인드 영역 내부)
        }
        continue;
      }

      vecs[j] = Eigen::Vector3d(pl[i+m].x, pl[i+m].y, pl[i+m].z);
      vecs[j] = vecs[j] - vec_a; // 이전 포인트에서 현재 포인트를 가리키는 벡터

      // O : 라이다 좌표 시스템의 원점
      // A : 현재 포인트
      // M, N : 이전/다음 포인트

      // |OA|*|MA| 하면 cos 각도 OAM 얻을 수 있다.
      
      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();

      // 이전 포인트가 normal point이고, 다음 포인트가 레이저 라인 위에 있고, (현재 포인트와 다음 포인트 사이의 거리) < 0.0225m 이고,
      // (현재 포인트와 다음 포인트 사이의 거리) > 4*(현재 포인트와 이전 포인트 사이의 거리) 일때
      // 이 edge point는 FIg. 7의 edge와 같다. (?)
      if(types[i].angle[j] < jump_up_limit) // cos(170)
      {
        types[i].edj[j] = Nr_180; // M은 OA를 확장한 선상에 있다.
      }
      else if(types[i].angle[j] > jump_down_limit) // cos(8)
      {
        types[i].edj[j] = Nr_zero; // M은 OA 위에 있다.
      }
    }

    types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();
    if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_zero && types[i].dista>0.0225 && types[i].dista>4*types[i-1].dista)
    {
      if(types[i].intersect > cos160) // 각도 MAN은 160 deg 보다 작아야 하고, 아니면 레이저에 평행(parallel) 하다.
      {
        if(edge_jump_judge(pl, types, i, Prev))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    // 위와 비슷하게,
    // 이전 포인트가 레이저 빔 위에 있고, 다음 포인트가 normal이고, (이전 포인트와 현재 포인트 사이의 거리) > 0.0225m 이고,
    // (이전 포인트와 현재 포인트 사이의 거리) > 4*(현재 포인트와 다음 포인트 사이의 거리) 일때
    else if(types[i].edj[Prev]==Nr_zero && types[i].edj[Next]== Nr_nor && types[i-1].dista>0.0225 && types[i-1].dista>4*types[i].dista)
    {
      if(types[i].intersect > cos160)
      {
        if(edge_jump_judge(pl, types, i, Next))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    // 이전 포인트가 normal point이고, (현재 포인트에서 중심까지의 거리) > 10m 이고, 다음 포인트가 블라인드 영역일 때
    else if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_inf)
    {
      if(edge_jump_judge(pl, types, i, Prev))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    // (현재 포인트에서 중심까지의 거리) > 10m 이고, 이전 포인트가 블라인드 영역이고, 다음 포인트가 normal point일 때
    else if(types[i].edj[Prev]==Nr_inf && types[i].edj[Next]==Nr_nor)
    {
      if(edge_jump_judge(pl, types, i, Next))
      {
        types[i].ftype = Edge_Jump;
      }
     
    }
    // 이전, 다음 포인트가 둘 다 normal이 아닐때
    else if(types[i].edj[Prev]>Nr_nor && types[i].edj[Next]>Nr_nor)
    {
      if(types[i].ftype == Nor)
      {
        types[i].ftype = Wire; // 사용하지 않는다. 작은 선분이나 불필요한 포인트로 취급.
      }
    }
  }

  plsize2 = plsize-1;
  double ratio;
  // plane feature 찾기
  for(uint i=head+1; i<plsize2; i++)
  {
    // 이전, 현재, 다음 포인트는 블라인드 영역 밖에 있다.
    if(types[i].range<blind || types[i-1].range<blind || types[i+1].range<blind)
    {
      continue;
    }
    
    // 이전과 현재 포인트 사이의 거리, 현재와 다음 포인트 사이의 거리는 너무 가까울 수 없다.
    if(types[i-1].dista<1e-8 || types[i].dista<1e-8)
    {
      continue;
    }
    // 남은 normal points가 꽤 있으므로 plane featrue 를 계속 찾는다.

    if(types[i].ftype == Nor)
    {
      // point-to-point spacing ratio = (large spacing)/(small spacing) 비율 계산
      if(types[i-1].dista > types[i].dista)
      {
        ratio = types[i-1].dista / types[i].dista;
      }
      else
      {
        ratio = types[i].dista / types[i-1].dista;
      }
      
      // 각도가 172.5 deg 보다 크고 ratio < 1.2 이면,
      if(types[i].intersect<smallp_intersect && ratio < smallp_ratio)
      {
        // 이전, 현재, 다음 포인트를 plane feature로 간주
        if(types[i-1].ftype == Nor)
        {
          types[i-1].ftype = Real_Plane;
        }
        if(types[i+1].ftype == Nor)
        {
          types[i+1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;
      }
    }
  }

  // plane points 저장
  int last_surface = -1;
  for(uint j=head; j<plsize; j++)
  {
    // possible plane points, 구한 plane points에 대해
    if(types[j].ftype==Poss_Plane || types[j].ftype==Real_Plane)
    {
      if(last_surface == -1)
      {
        last_surface = j;
      }

      // 일반적으로 한 행에 몇몇 선이 있다. (?)
      // 샘플링 주기의 plane points를 사용 (새로운 표면 포인트를 찾았을 때부터 몇 개의 포인트마다 하나씩 indifference filter 적용)
      if(j == uint(last_surface+point_filter_num-1))
      {
        PointType ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.intensity = pl[j].intensity;
        ap.curvature = pl[j].curvature;
        pl_surf.push_back(ap);

        last_surface = -1;
      }
    }
    else
    {
      // 평면 위 edge 포인트 (?)
      if(types[j].ftype==Edge_Jump || types[j].ftype==Edge_Plane)
      {
        pl_corn.push_back(pl[j]);
      }
      // indiscriminate filtering으로 이전에 찾은 표면 포인트들이 edge에 도달한 경우
      if(last_surface != -1)
      {
        PointType ap;
        // 이전 표면 포인트와 이번 edge line 사이의 모든 포인트의 중력 중심을 얻고 표면 포인트에 저장
        for(uint k=last_surface; k<j; k++)
        {
          ap.x += pl[k].x;
          ap.y += pl[k].y;
          ap.z += pl[k].z;
          ap.intensity += pl[k].intensity;
          ap.curvature += pl[k].curvature;
        }
        ap.x /= (j-last_surface);
        ap.y /= (j-last_surface);
        ap.z /= (j-last_surface);
        ap.intensity /= (j-last_surface);
        ap.curvature /= (j-last_surface);
        pl_surf.push_back(ap);
      }
      last_surface = -1;
    }
  }
}

// topic command. 사용하지 않음
void Preprocess::pub_func(PointCloudXYZI &pl, const ros::Time &ct)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "livox";
  output.header.stamp = ct;
}

// plane 판단
int Preprocess::plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct)
{
  double group_dis = disA*types[i_cur].range + disB; // 0.01*sqrt(x^2+y^2)+0.1 은 일반적으로 0.1에 가까우며 100m에 도달했을 때만 0.2의 값을 가진다.
  group_dis = group_dis * group_dis;
  // i_nex = i_cur;

  double two_dis;
  vector<double> disarr; // 이전고 ㅏ다음 포인트 사이의 거리에 대한 배열
  disarr.reserve(20);

  // 거리가 짧고 포인트가 서로 가깝다. 8개의 포인트를 얻는다.
  for(i_nex=i_cur; i_nex<i_cur+group_size; i_nex++)
  {
    if(types[i_nex].range < blind)
    {
      curr_direct.setZero(); // 라이다 원점으로부터 거리가 매우 짧다. 법선 벡터를 영벡터로 설정한다.
      return 2;
    }
    disarr.push_back(types[i_nex].dista); // 현재와 다음 포인트 사이의 거리를 저장한다.
  }
  
  // 연속적인 포인트들이 조건을 만족하는지 확인
  for(;;)
  {
    if((i_cur >= pl.size()) || (i_nex >= pl.size())) break; // 인덱스가 총 포인트 수를 초과하면 break

    if(types[i_nex].range < blind)
    {
      curr_direct.setZero(); // 라이다 원점으로부터 거리가 매우 짧다. 법선 벡터를 영벡터로 설정한다.
      return 2;
    }
    // 이전 i_nex 포인트에서 i_cur 포인트까지의 거리
    vx = pl[i_nex].x - pl[i_cur].x;
    vy = pl[i_nex].y - pl[i_cur].y;
    vz = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx*vx + vy*vy + vz*vz;
    if(two_dis >= group_dis) // i_cur 포인트에서부터의 거리가 너무 멀면 break
    {
      break;
    }
    disarr.push_back(types[i_nex].dista); // 현재 포인트에서 다음 포인트까지의 거리를 저장
    i_nex++;                              // i_nex + 1
  }

  double leng_wid = 0;
  double v1[3], v2[3];
  for(uint j=i_cur+1; j<i_nex; j++)
  {
    if((j >= pl.size()) || (i_cur >= pl.size())) break;

    // i_cur 포인트를 A라고 하고, j 포인트를 B라고 하고, i_nex 포인트를 C라고 한다.
    // 벡터 AB
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    // 벡터 AB와 벡터 AC의 cross product
    v2[0] = v1[1]*vz - vy*v1[2];
    v2[1] = v1[2]*vx - v1[0]*vz;
    v2[2] = v1[0]*vy - vx*v1[1];
    // 물리적 의미 : 평행사변형 ABC의 면적의 제곱 (|AC|*h, B에서 직선 AC 까지의 거리 h)
    // 
    double lw = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
    if(lw > leng_wid)
    {
      leng_wid = lw; // 가장 큰 영역의 제곱 (즉, AC에서 가장 먼 B를 찾는다.)
    }
  }

  // |AC|*|AC|/(|AC|*|AC|*h*h)<225
  // 즉, h>1/15, 포인트 B에서 AC까지의 거리가 0.06667m 보다 크면
  // 너무 가까워서 plane에 fit할 수 없다.
  if((two_dis*two_dis/leng_wid) < p2l_ratio)
  {
    curr_direct.setZero(); // 라이다 원점으로부터 거리가 매우 짧다. 법선 벡터를 영벡터로 설정한다.
    return 0;
  }

  // 두 포인트 사이의 거리를 큰 값에서 작은 값 순서로 정렬
  uint disarrsize = disarr.size();
  for(uint j=0; j<disarrsize-1; j++)
  {
    for(uint k=j+1; k<disarrsize; k++)
    {
      if(disarr[j] < disarr[k])
      {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  // 가장 가까운 포인트는 아직도 너무 가까울 것이다.
  if(disarr[disarr.size()-2] < 1e-16)
  {
    curr_direct.setZero();
    return 0;
  }

  // AVIA에 대해서 따로 처리 (주석 단 사람은 위의 judgment 방식으로 충분히 할 수 있을 것 같은데 왜 따로 처리하는지 모르겠다고 함)
  if(lidar_type==AVIA)
  {
    // 포인트 변화가 너무 크면 레이저 빔에 평행한 것일 수 있으므로 버림
    double dismax_mid = disarr[0]/disarr[disarrsize/2];
    double dismid_min = disarr[disarrsize/2]/disarr[disarrsize-2];

    if(dismax_mid>=limit_maxmid || dismid_min>=limit_midmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  else
  {
    double dismax_min = disarr[0] / disarr[disarrsize-2];
    if(dismax_min >= limit_maxmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  
  curr_direct << vx, vy, vz;
  curr_direct.normalize(); // 법선 벡터 normalization
  return 1;
}

// marginal judgment
bool Preprocess::edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir)
{
  if(nor_dir == 0)
  {
    if(types[i-1].range<blind || types[i-2].range<blind) // 처음 두 포인트는 블라인드 영역에 있을 수 없다.
    {
      return false;
    }
  }
  else if(nor_dir == 1)
  {
    if(types[i+1].range<blind || types[i+2].range<blind) // 마지막 두 포인트는 블라인드 영역에 있을 수 없다.
    {
      return false;
    }
  }
  // i-2와 i-1, i와 i+1 두 경우에 대해 포인트 간의 거리에 대한 판단
  double d1 = types[i+nor_dir-1].dista;
  double d2 = types[i+3*nor_dir-2].dista;
  double d;

  // 이전의 큰 것과 다음의 작은 것을 바꾼다.
  if(d1<d2)
  {
    d = d1;
    d1 = d2;
    d2 = d;
  }

  d1 = sqrt(d1);
  d2 = sqrt(d2);

 
  if(d1>edgea*d2 || (d1-d2)>edgeb)
  {
    // 거리가 너무 멀면, 가려질 수 있으므로 edge 포인트로 사용하지 않는다.
    return false;
  }
  
  return true;
}
