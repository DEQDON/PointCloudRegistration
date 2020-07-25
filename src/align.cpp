/*
* This Program is the toy system implementation of a point cloud alignment algorithm
* using iterative closest point algorithm. 
* I use Hamilton quaternion for parameterization.
* Press 'space' for a new iteration.
* 
* Point Cloud operation references to 
* http://pointclouds.org/documentation/tutorials/interactive_icp.html#interactive-icp
*/


#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/common/transforms.h>     // pcl::transformPointCloud function

#include <pcl/console/time.h>          // TicToc

#include <Eigen/Eigenvalues>


bool next_iteration = false;

Eigen::Matrix3f asymmetric(Eigen::Vector3f vec);

Eigen::Matrix4f leftMulti(Eigen::Vector4f vec);
Eigen::Matrix4f leftMulti(Eigen::Quaternionf q);
Eigen::Matrix4f rightMulti(Eigen::Vector4f vec);
Eigen::Matrix4f rightMulti(Eigen::Quaternionf q);

Eigen::Matrix4f align(pcl::PointCloud<pcl::PointXYZI>::Ptr pc1, pcl::PointCloud<pcl::PointXYZI>::Ptr pc2,
pcl::PointCloud<pcl::PointXYZI>::Ptr transformedPC2);

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event, void* nothing);

 
int main(int argc, char *argv[])
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr  cloud_1 (new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr  cloud_2 (new pcl::PointCloud<pcl::PointXYZI>);
	if (pcl::io::loadPCDFile<pcl::PointXYZI>("pcdData/first.pcd", *cloud_1) == -1) {
		std::cout << "Cloud 1 reading failed." << std::endl;
		return -1;
	}
	if (pcl::io::loadPCDFile<pcl::PointXYZI>("pcdData/second.pcd", *cloud_2) == -1) {
		std::cout << "Cloud 2 reading failed." << std::endl;
		return -1;
	}

	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_1 (new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_2 (new pcl::PointCloud<pcl::PointXYZI>);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sorfilter_1 (true); // Initializing with true will allow us to extract the removed indices
	sorfilter_1.setInputCloud (cloud_1);
	sorfilter_1.setMeanK (8);
	sorfilter_1.setStddevMulThresh (1.0);
	sorfilter_1.filter (*cloud_filtered_1);
	// The resulting cloud_out contains all points of cloud_in that have an average distance to their 8 nearest neighbors that is below the computed threshold
	// Using a standard deviation multiplier of 1.0 and assuming the average distances are normally distributed there is a 84.1% chance that a point will be an inlier
	// indices_rem_1 = sorfilter.getRemovedIndices ();
	// The indices_rem array indexes all points of cloud_in that are outliers

	pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sorfilter_2 (true); // Initializing with true will allow us to extract the removed indices
	sorfilter_2.setInputCloud (cloud_2);
	sorfilter_2.setMeanK (8);
	sorfilter_2.setStddevMulThresh (1.0);
	sorfilter_2.filter (*cloud_filtered_2);
	
	pcl::PointCloud<pcl::PointXYZI>::Ptr  latest_cloud_2(new pcl::PointCloud<pcl::PointXYZI>);
	*latest_cloud_2 = *cloud_filtered_2;

	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);

	// Cloud 1 is red. Cloud 2 is white.
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> cloud_1_color(cloud_filtered_1, 199, 25, 33);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> cloud_2_color(cloud_filtered_2, 255, 255, 255);

	viewer->addPointCloud<pcl::PointXYZI> (cloud_filtered_1, cloud_1_color, "first");
  	viewer->addPointCloud<pcl::PointXYZI> (cloud_filtered_2, cloud_2_color, "second");

	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "first");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "second");
	viewer->addCoordinateSystem (1.0);
	viewer->setCameraPosition (-3.68332, 2.94092, 50.71266, 0.289847, 0.921947, 30);

	int iterations = 0;

	// Register keyboard callback :
	viewer->registerKeyboardCallback (&keyboardEventOccurred, (void*) NULL);
	// Register timer;
	pcl::console::TicToc time;

	// Display the visualiser
	while (!viewer->wasStopped())
	{
		viewer->spinOnce();

		// The user pressed "space" :
		if (next_iteration) {
			// The Iterative Closest Point algorithm
			pcl::PointCloud<pcl::PointXYZI>::Ptr  transformed_cloud_2(new pcl::PointCloud<pcl::PointXYZI>);
			*transformed_cloud_2 = *latest_cloud_2;
			time.tic();
			align(cloud_filtered_1, latest_cloud_2, transformed_cloud_2);
			std::cout << "Applied ICP iteration #" << ++iterations << " in " << time.toc() << " ms" << std::endl << std::endl;
			viewer->updatePointCloud(transformed_cloud_2, cloud_2_color, "second");
			*latest_cloud_2 = *transformed_cloud_2;
		}

		next_iteration = false;
	}
 
	return 0;
}


Eigen::Matrix4f align(pcl::PointCloud<pcl::PointXYZI>::Ptr pc1, pcl::PointCloud<pcl::PointXYZI>::Ptr pc2,
pcl::PointCloud<pcl::PointXYZI>::Ptr transformedPC2)
{
	// The index of points in cloud 2 corresponding to points in cloud 1.
  	std::vector<int> vPointIdxNNCloud2;
	// The index of points in cloud 1 corresponding to points in cloud 2.
  	std::vector<int> vPointIdxNNCloud1;
	pcl::PointXYZI searchPoint;
	std::vector<int> pointIdxNKNSearch(1);
	std::vector<float> pointNKNSquaredDistance(1);
	
	std::vector<Eigen::Vector3f> vPoints_1;
	std::vector<Eigen::Vector3f> vPoints_2;
	std::vector<float> vIntensity_1;
	std::vector<float> vIntensity_2;

	pcl::KdTreeFLANN<pcl::PointXYZI> kdtree_2;
  	kdtree_2.setInputCloud (pc2);

	float omega = 0.0;

	// Nearest neighbor matching.
	for (int i = 0; i < pc1->size(); i++) {
		searchPoint.x = pc1->points[i].x;
		searchPoint.y = pc1->points[i].y;
		searchPoint.z = pc1->points[i].z;
		kdtree_2.nearestKSearch (searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
		if (pointNKNSquaredDistance[0] < 1.0) {
			int idx2 = pointIdxNKNSearch[0];
			vPointIdxNNCloud1.push_back(i);
			vPointIdxNNCloud2.push_back(idx2);
			Eigen::Vector3f vec1(searchPoint.x, searchPoint.y, searchPoint.z);
			Eigen::Vector3f vec2(pc2->points[idx2].x, pc2->points[idx2].y, pc2->points[idx2].z);
			vPoints_1.push_back(vec1);
			vPoints_2.push_back(vec2);
			vIntensity_1.push_back(pc1->points[i].intensity);
			vIntensity_2.push_back(pc2->points[idx2].intensity);

			omega += pc1->points[i].intensity;

			//std::cout << "point #1 coordinate:" << std::endl << vec1 << std::endl << std::endl;
			//std::cout << "point #2 coordinate:" << std::endl << vec2 << std::endl << std::endl;

			//pcl::visualization::createSphere(Eigen::Vector4f(vec1(0), vec1(1), vec1(2), 0), 5);
			//pcl::visualization::createSphere(Eigen::Vector4f(vec2(0), vec2(1), vec2(2), 0), 5);
		}
	}
	
	Eigen::Vector3f _p(0, 0, 0);
	Eigen::Vector3f _y(0, 0, 0);
	Eigen::Matrix4f W = Eigen::Matrix4f::Zero();
	//std::cout << W << std::endl;

	for (int i = 0; i < vPoints_1.size(); i++) {
		Eigen::Vector3f p_j = vPoints_1[i];
		Eigen::Vector3f y_j = vPoints_2[i];
		float omega_j = vIntensity_1[i];
		_p += omega_j * p_j / omega;
		_y += omega_j * y_j / omega;
		
	}
	Eigen::Vector4f p(0, _p(0), _p(1), _p(2));
	Eigen::Vector4f y(0, _y(0), _y(1), _y(2));

	std::cout << std::endl << "***************** A new iteration begins ******************" << std::endl;

	std::cout << "p = " << std::endl << p << std::endl << std::endl;
	std::cout << "y = " << std::endl << y << std::endl << std::endl;

	for (int i = 0; i < vPoints_1.size(); i++) {
		Eigen::Vector4f p_j(0, vPoints_1[i](0), vPoints_1[i](1), vPoints_1[i](2));
		Eigen::Vector4f y_j(0, vPoints_2[i](0), vPoints_2[i](1), vPoints_2[i](2));
		float omega_j = vIntensity_1[i];

		auto A = rightMulti(y_j - y) - leftMulti(p_j - p);
		W += omega_j * A.transpose() * A / omega;
	}

	std::cout << "W = " << std::endl << W << std::endl << std::endl;

	Eigen::EigenSolver<Eigen::Matrix4f> es(W);
	Eigen::Matrix4f D = es.pseudoEigenvalueMatrix();
	Eigen::Matrix4f V = es.pseudoEigenvectors();
	std::cout << "The pseudo-eigenvalue matrix D is:" << std::endl << D << std::endl << std::endl;
	std::cout << "The pseudo-eigenvector matrix V is:" << std::endl << V << std::endl << std::endl;
	//std::cout << "Finally, V * D * V^(-1) = " << std::endl << V * D * V.inverse() << std::endl;

	//int col_index, row_index;
	//std::cout << D.minCoeff(&row_index, &col_index) << std::endl;
	//std::cout << row_index << " " << col_index << std::endl;

	// ************** Attention! ***************
	// Eigen Quaternion initialization has this pitfall!
	// Please always use "Eigen::Quaternionf q(w, x, y, z);" to avoid the pitfall!
	// If I use "Eigen::Quaternionf q(V.col(0));" here, the quaternion would be
	// "q(V(1, 0), V(2, 0), V(3, 0), V(0, 0));"

	Eigen::Quaternionf q(V(0, 0), V(1, 0), V(2, 0), V(3, 0));
	
	std::cout << "The quaternion (before normalization) is:" << std::endl << q.coeffs() << std::endl;
	std::cout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << std::endl;
	q.normalize();
	std::cout << "The quaternion is:" << std::endl << q.coeffs() << std::endl << std::endl;
	std::cout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << std::endl;
	std::cout << "The rotation matrix is:" << std::endl << q.toRotationMatrix() << std::endl << std::endl;


	Eigen::Quaternionf qyq_ = q * Eigen::Quaternionf(y(0), y(1), y(2), y(3)) * q.inverse();
	std::cout << "p =" << std::endl << p << std::endl << std::endl;
	std::cout << "q * y * q^-1 =" << std::endl << qyq_.coeffs() << std::endl << std::endl;
	Eigen::Vector3f r = (p - Eigen::Vector4f(qyq_.w(), qyq_.x(), qyq_.y(), qyq_.z())).tail(3);
	std::cout << "The translation is:" << std::endl << r << std::endl << std::endl;

	Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
	T.topLeftCorner(3, 3) = q.toRotationMatrix();
	T.topRightCorner(3, 1) = r;

	//std::cout << "The rotation matrix is:" << std::endl << q.toRotationMatrix() << std::endl << std::endl;
	//std::cout << "The translation is:" << std::endl << r << std::endl << std::endl;

	//Eigen::Matrix3f R_vec = q.w() * Eigen::Matrix3f::Identity() + 2 * q.w() * asymmetric(q.vec())
	                        //+ asymmetric(q.vec()) * asymmetric(q.vec()) + q.vec() * q.vec().transpose();
	//std::cout << "The rotation matrix by vector calculation is:" <<
	 //std::endl << R_vec << std::endl << std::endl;

	std::cout << "The transformation matrix is:" << std::endl << T.matrix() << std::endl << std::endl;
	pcl::transformPointCloud (*pc2, *transformedPC2, T);

}

Eigen::Matrix3f asymmetric(Eigen::Vector3f vec)
{
	Eigen::Matrix3f as = Eigen::Matrix3f::Zero();
	as(0, 1) = -vec(2);
	as(0, 2) = vec(1);
	as(1, 0) = vec(2);
	as(1, 2) = -vec(0);
	as(2, 0) = -vec(1);
	as(2, 1) = vec(0);

	//std::cout << "asymmetric test:" << std::endl <<
	//vec << std::endl << as << std::endl << std::endl;

	return as;
}

Eigen::Matrix4f leftMulti(Eigen::Vector4f vec)
{
	Eigen::Matrix4f left = Eigen::Matrix4f::Zero();
	left.topRightCorner<1, 3>() = -vec.tail(3).transpose();
	left.bottomLeftCorner<3, 1>() = vec.tail(3);
	left.bottomRightCorner<3, 3>() = asymmetric(vec.tail(3));
	left(0, 0) += vec(0);
	left(1, 1) += vec(0);
	left(2, 2) += vec(0);
	left(3, 3) += vec(0);

	return left;
}

Eigen::Matrix4f rightMulti(Eigen::Vector4f vec)
{
	Eigen::Matrix4f right = Eigen::Matrix4f::Zero();
	right.topRightCorner<1, 3>() = -vec.tail(3).transpose();
	right.bottomLeftCorner<3, 1>() = vec.tail(3);
	right.bottomRightCorner<3, 3>() = -asymmetric(vec.tail(3));
	right(0, 0) += vec(0);
	right(1, 1) += vec(0);
	right(2, 2) += vec(0);
	right(3, 3) += vec(0);

	return right;
}

Eigen::Matrix4f leftMulti(Eigen::Quaternionf q)
{
	Eigen::Matrix4f left = Eigen::Matrix4f::Zero();
	left.topRightCorner<1, 3>() = -q.vec();
	left.bottomLeftCorner<3, 1>() = q.vec();
	left.bottomRightCorner<3, 3>() = asymmetric(q.vec());
	left(0, 0) += q.w();
	left(1, 1) += q.w();
	left(2, 2) += q.w();
	left(3, 3) += q.w();

	return left;
}

Eigen::Matrix4f rightMulti(Eigen::Quaternionf q)
{
	Eigen::Matrix4f right = Eigen::Matrix4f::Zero();
	right.topRightCorner<1, 3>() = -q.vec();
	right.bottomLeftCorner<3, 1>() = q.vec();
	right.bottomRightCorner<3, 3>() = -asymmetric(q.vec());
	right(0, 0) += q.w();
	right(1, 1) += q.w();
	right(2, 2) += q.w();
	right(3, 3) += q.w();

	return right;
}

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event, void* nothing)
{
  if (event.getKeySym () == "space" && event.keyDown ())
    next_iteration = true;
}