﻿#include "backend/event_pano_warper.h"
#include "backend/event_pano_warper_cuda.h"
#include <sensor_msgs/Imu.h>
#include <opencv2/core/eigen.hpp>
#include <glog/logging.h>
#include <chrono>

namespace cmax_slam
{

    void EventWarper::initialize(int camera_width, int camera_height,
                                 std::vector<cv::Point3d> *precomputed_bearing_vectors_ptr)
    {
        // Set camera info
        sensor_width_ = camera_width;
        sensor_height_ = camera_height;

        // Set pano map
        pano_width_ = map_opt_.pano_width;
        pano_height_ = map_opt_.pano_height;
        pano_size_ = cv::Size(pano_width_, pano_height_);

        // States in the backend
        IG_ = cv::Mat::zeros(pano_size_, CV_32FC1);
        IGp_ = cv::Mat::zeros(pano_size_, CV_32FC1);
        IL_ = cv::Mat(pano_size_, CV_32FC1);
        IL_old_ = cv::Mat::zeros(pano_size_, CV_32FC1);
        IL_new_ = cv::Mat::zeros(pano_size_, CV_32FC1);

        // Initialize the map of updating time for each pixel
        IG_update_times_map_ = cv::Mat::zeros(pano_size_, CV_8UC1);

        // Equirectangular projection
        pano_cam_ = dvs::EquirectangularCamera(pano_size_, 360.0, 180.0);
        precomputed_bearing_vectors_ = precomputed_bearing_vectors_ptr;

        VLOG(2) << "Event warper initialized";

        // Initialize wrapper 
        double fx, fy;
        pano_cam_.getScalingFactors(fx, fy);
        Eigen::Vector2d center;
        pano_cam_.getCenter(center);
        cuda_event_warper.setSubsetSize(200);
        cuda_event_warper.setWarpingParameters(fx, fy, center[0], center[1]);
        cuda_event_warper.setILOldNewSize(512, 1024, 512, 1024);
        cuda_event_warper.mallocDeviceMemory();
        cuda_event_warper.mallocHostMemory();
        cuda_event_warper.checkInitializedSafely();

    }

    void EventWarper::warpEventToMap(const cv::Point2i &pt_in,
                                     const Sophus::SO3d &rot,
                                     cv::Point2d &pt_out)
    {
        const int idx = pt_in.y * sensor_width_ + pt_in.x;

        // Get bearing vector (in look-up-table) corresponding to current event's pixel
        const cv::Point3d bvec = precomputed_bearing_vectors_->at(idx);
        Eigen::Vector3d e_ray_cam(bvec.x, bvec.y, bvec.z);

        // Rotate according to pose(t)
        Eigen::Matrix3d R = rot.matrix();
        Eigen::Vector3d e_ray_w = R * e_ray_cam;

        // Project onto the panoramic map
        Eigen::Vector2d px_mosaic = pano_cam_.projectToImage(e_ray_w, nullptr);
        pt_out = cv::Point2d(px_mosaic[0], px_mosaic[1]);
    }

    void EventWarper::drawSensorFOV(cv::Mat &canvas,
                                    const Sophus::SO3d &R,
                                    const cv::Vec3i &color)
    {
        for (int x = 0; x < sensor_width_; ++x)
        {
            cv::Point2d warped_pt;
            warpEventToMap(cv::Point2i(x, 0), R, warped_pt);
            canvas.at<cv::Vec3b>(warped_pt) = color;

            warpEventToMap(cv::Point2i(x, sensor_height_ - 1), R, warped_pt);
            canvas.at<cv::Vec3b>(warped_pt) = color;
        }

        for (int y = 0; y < sensor_height_; ++y)
        {
            cv::Point2d warped_pt;
            warpEventToMap(cv::Point2i(0, y), R, warped_pt);
            canvas.at<cv::Vec3b>(warped_pt) = color;

            warpEventToMap(cv::Point2i(sensor_width_ - 1, y), R, warped_pt);
            canvas.at<cv::Vec3b>(warped_pt) = color;
        }
    }

    void EventWarper::setUpdateTimesIG(const Sophus::SO3d &rot,
                                       const int radius)
    {
        cv::Mat mask = cv::Mat::zeros(IG_.rows, IG_.cols, CV_8UC1);
        for (int x = 0; x < sensor_width_; ++x)
        {
            for (int y = 0; y < sensor_height_; ++y)
            {
                cv::Point2d warped_pt;
                warpEventToMap(cv::Point2i(x, y), rot, warped_pt);
                const int ic = warped_pt.x, ir = warped_pt.y;
                for (int i = -radius; i <= radius; i++)
                {
                    for (int j = -radius; j <= radius; j++)
                    {
                        int x_mask = ic + i, y_mask = ir + j;
                        if (0 <= y_mask + j && y_mask < IG_.rows && 0 <= x_mask && x_mask < IG_.cols)
                        {
                            mask.at<uchar>(y_mask, x_mask) = 1;
                        }
                    }
                }
            }
        }

        cv::add(IG_update_times_map_, mask, IG_update_times_map_);
    }

    void EventWarper::updateIG()
    {
        VLOG(2) << "Update Global IWE to prepare for the next optimization";

        /* Stop updating at some point */
        // Check the updating times of each pixel,
        // stop accmulating events if it reach the maximal times of updating
        for (int y = 0; y < IG_.rows; y++)
        {
            for (int x = 0; x < IG_.cols; x++)
            {
                if (IG_update_times_map_.at<uchar>(y, x) <= map_opt_.max_update_times)
                {
                    IG_.at<float>(y, x) += IL_old_.at<float>(y, x);
                }
            }
        }
    }

    void EventWarper::updateIGp()
    {
        /* Linear scale */
        IG_.copyTo(IGp_);
    }

    void EventWarper::updateAlpha()
    {
        /* Compute event density using the image area proposed in Gallego et al. CVPR'19 */
        if (cv::countNonZero(IGp_) < 1)
        {
            alpha_ = 0;
            return;
        }
        /// (1) event_density(IGp)
        // Aggregated support of the IWE (to be minimized)
        // Decaying exponential weight of the level-dependent supports
        const float lambda0 = 1.0f;
        cv::Mat IGp_exp;
        cv::exp(-(1.0 / lambda0) * IGp_, IGp_exp);
        cv::Mat IGp_integrand = 1.f - IGp_exp;
        const double area_IGp = cv::sum(IGp_integrand)[0];
        const double num_ev_IGp = cv::sum(IGp_)[0];
        const double ev_density_IGp = num_ev_IGp / area_IGp;

        /// (2) event_density(IL)
        cv::Mat IL_exp;
        cv::exp(-(1.0 / lambda0) * IL_, IL_exp);
        cv::Mat IL_integrand = 1.f - IL_exp;
        const double area_IL = cv::sum(IL_integrand)[0];
        const double num_ev_IL = cv::sum(IL_)[0];
        const double ev_density_IL = num_ev_IL / area_IL;

        /// (3) compute alpha
        alpha_ = ev_density_IL / ev_density_IGp;

        VLOG(2) << "[EventWarper::updateAlpha] alpha = " << alpha_ << std::endl;
    }

    void EventWarper::computeImageOfWarpedEvents(Trajectory *traj,
                                                 std::vector<dvs_msgs::Event> *event_subset,
                                                 cv::Mat *iwe,
                                                 std::vector<cv::Mat> *iwe_deriv)
    {
        // Create image of warped events);
        IL_old_.setTo(0);
        IL_new_.setTo(0);
        cuda_event_warper.resetToZeroILOldNew();
        if (iwe_deriv != nullptr)
        {
            // Create images of the derivative of the warped events wrt pose parameters
            const int num_bands = 3 * (traj->size() - num_cps_fixed_);
            iwe_deriv->clear();
            for (int i = 0; i < num_bands; i++)
            {
                iwe_deriv->push_back(cv::Mat::zeros(pano_size_, CV_32FC1));
            }
        }

        // Get event_batch using two cursors
        for (auto ev_batch_beg = event_subset->begin();
             ev_batch_beg < event_subset->end() - 1;
             ev_batch_beg += warp_opt_.event_batch_size)
        {
            int num_events_left = event_subset->end() - ev_batch_beg;
            auto ev_batch_end = (num_events_left > warp_opt_.event_batch_size) ? ev_batch_beg + warp_opt_.event_batch_size : event_subset->end();
            warpAndAccumulateEvents(traj, ev_batch_beg, ev_batch_end, iwe_deriv);
        }

        // Combine the old and new parts to get the whole IL for the current time window
        cv::add(IL_old_, IL_new_, IL_);

        if (first_iter_)
        {
            // Update IG'
            updateIGp();

            // Compute alpha for the first time window
            updateAlpha();

            first_iter_ = false;
        }

        // I = IL + alpha * IG'
        cv::scaleAdd(IGp_, alpha_, IL_, *iwe);

        // Smooth the image (to spread the votes)
        // For speed-up, smoothing may not be used, since bilinear voting has been implemented.
        if (warp_opt_.blur_sigma > 0)
        {
            cv::GaussianBlur(*iwe, *iwe, cv::Size(0, 0), warp_opt_.blur_sigma);
            if (iwe_deriv != nullptr)
            {
                const int num_bands = 3 * (traj->size() - num_cps_fixed_);
                CHECK_EQ(num_bands, iwe_deriv->size());
                for (int i = 0; i < num_bands; i++)
                {
                    cv::GaussianBlur(iwe_deriv->at(i), iwe_deriv->at(i),
                                     cv::Size(0, 0), warp_opt_.blur_sigma);
                }
            }
        }
    }

    void EventWarper::warpAndAccumulateEvents(Trajectory *traj,
                                              std::vector<dvs_msgs::Event>::iterator event_begin,
                                              std::vector<dvs_msgs::Event>::iterator event_end,
                                              std::vector<cv::Mat> *iwe_deriv)
    {
        auto start = std::chrono::high_resolution_clock::now();
        static double totalTime = 0; // Static variable to accumulate total time
        static int numRuns = 0;      // Static variable to keep track of the number of runs

        // Share a common pose for all events in the batch
        ros::Time time_first = event_begin->ts;
        ros::Time time_last = (event_end - 1)->ts;
        ros::Duration time_dt = time_last - time_first;
        ros::Time time_batch = time_first + time_dt * 0.5;

        Sophus::SO3d so3;

        // All events in the same batch share some common derivatives:
        // ddrot_ddrot_cp: (pertubation on the poses [drot] w.r.t. perturbations on the control poses [drot_cp])
        cv::Mat ddrot_ddrot_cp;
        int idx_cp_beg;
        if (iwe_deriv != nullptr)
        {
            so3 = traj->evaluate(time_batch, &idx_cp_beg, &ddrot_ddrot_cp);
        } // 3X1
        else
        {
            so3 = traj->evaluate(time_batch, nullptr, nullptr);
        }

        // Convert to cv::Mat
        Eigen::Matrix3d R_eigen = so3.matrix();
        cv::Matx33d R_cv;
        cv::eigen2cv(R_eigen, R_cv);

        cv::Mat jac_warped_pt_wrt_traj;
        if (iwe_deriv != nullptr)
        {
            jac_warped_pt_wrt_traj = cv::Mat::zeros(2, 3 * traj->NumInvolvedControlPoses(), CV_32FC1);
        }

        // Prepare memories
        int subset_size = std::distance(event_begin, event_end);
        bool *h_oldevent = (bool *)malloc(subset_size * sizeof(bool));


        // Get the roatted event ray from bearing vectors and check if event is out of date too
        int num_old_events = 0;
        int num_new_events = 0;
        // LOG(INFO) << "Looping thru subset" << std::endl;
        cuda_event_warper.setSubsetSize(subset_size);

        for (int i = 0; i < subset_size; i++)
        {
            auto ev = event_begin + i;
            const cv::Point3d bvec = precomputed_bearing_vectors_->at(ev->y * sensor_width_ + ev->x);

            // Rotate according to pose(t)
            Eigen::Vector3d e_ray_cam(bvec.x, bvec.y, bvec.z);
            Eigen::Vector3d e_ray_w = R_eigen * e_ray_cam; // Not worth
            // float h_R[9];
            // float h_vec[3] = {(float)bvec.x, (float)bvec.y, (float)bvec.z};
            // float h_result[3];

            // for (int i = 0; i < 3; ++i)
            // {
            //     for (int j = 0; j < 3; ++j)
            //     {
            //         h_R[i * 3 + j] = static_cast<float>(R_eigen(i, j));
            //     }
            // }

            // matrixVectorMultWrapper(h_R, h_vec, h_result);

            // h_e_ray_rotated[i].x = h_result[0];
            // h_e_ray_rotated[i].y = h_result[1];
            // h_e_ray_rotated[i].z = h_result[2];
            cuda_event_warper.updateEventRotatedRayArr(i, e_ray_w[0], e_ray_w[1], e_ray_w[2]);

            if (ev->ts < t_next_win_beg_)
            {
                h_oldevent[i] = true;
                num_old_events += 1;
            }
            else
            {
                h_oldevent[i] = false;
                num_new_events += 1;
            }
        }

        // LOG(INFO) << "Number of old events:" << num_old_events << ", Number of new events:" << num_new_events;
        cuda_event_warper.warpEventsWrapper();
        // LOG(INFO) << "Finished calling warpEventsWrapper" << std::endl;

        float2 *h_warped_pixel_pos = cuda_event_warper.getWarpedPixelPosPtr();

        // Split warped pano pixel popsitions to new and old to be appended on diff IL respectively
        float2 *h_new_warped_pixel_pos;
        // int2 *h_new_x_y;
        // float2 *h_new_dx_dy;

        float2 *h_old_warped_pixel_pos;
        // int2 *h_old_x_y;
        // float2 *h_old_dx_dy;

        if (num_new_events > 0)
        {
            h_new_warped_pixel_pos = (float2 *)malloc(num_new_events * sizeof(float2));
        }

        if (num_old_events > 0)
        {
            h_old_warped_pixel_pos = (float2 *)malloc(num_old_events * sizeof(float2));
        }

        // Split warped pixel position to old and new
        int old_counter = 0;
        int new_counter = 0;
        for (int i = 0; i < subset_size; i++)
        {
            if (h_oldevent[i] == true)
            {
                h_old_warped_pixel_pos[old_counter] = h_warped_pixel_pos[i];
                old_counter++;
            }
            else
            {
                h_new_warped_pixel_pos[new_counter] = h_warped_pixel_pos[i];
                new_counter++;
            }
        }

        // Call cuda kernel
        cuda_event_warper.accumulatePolarityWrapper(h_new_warped_pixel_pos, h_old_warped_pixel_pos, num_new_events, num_old_events);
        // Copy the new data over
        float *IL_new_ptr = cuda_event_warper.getILNewPtr();
        float *IL_old_ptr = cuda_event_warper.getILOldPtr();

        for (int i = 0; i < IL_new_.rows; i++)
        {
            for (int j = 0; j < IL_new_.cols; j++)
            {
                int idx = (i * IL_new_.rows) + j;
                IL_new_.at<float>(i,j) = IL_new_ptr[idx];
            }
        }
        for (int i = 0; i < IL_old_.rows; i++)
        {
            for (int j = 0 ; j < IL_old_.cols; j++)
            {
                int idx = (i * IL_old_.rows) + j;
                IL_old_.at<float>(i,j) = IL_old_ptr[idx];
            }
        }

        if (num_new_events > 0)
        {
            free(h_new_warped_pixel_pos);
        }
        if (num_old_events > 0)
        {
            free(h_old_warped_pixel_pos);
        }

        free(h_oldevent);
        // Measure time at the end

        cuda_event_warper.resetToZeroXXYY(num_old_events, num_new_events);

        auto end = std::chrono::high_resolution_clock::now();

        // Calculate the elapsed time in milliseconds
        std::chrono::duration<double, std::milli> elapsed = end - start;
        totalTime += elapsed.count(); // Accumulate total time in milliseconds
        ++numRuns;                    // Increment the number of runs

        // Calculate and print the average time
        double averageTime = totalTime / numRuns;
        // std::cout << "Average run time after " << numRuns << " run(s): " << averageTime << " ms" << std::endl;
        std::cout << "elapsed " << elapsed.count() << " ms" << std::endl;
    }
}
