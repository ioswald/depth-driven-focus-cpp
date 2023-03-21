#include <iostream>

// include depthai library
#include <depthai/depthai.hpp>

// include opencv library (Optional, used only for the following example)
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;

dai::Pipeline CreatePipeline();

float disparityMultiplier;
	auto lensMin = 0;
	auto lensMax = 255;
	auto lensPos = 150;

int main() {
	dai::Device device(CreatePipeline());

	auto controlQ = device.getInputQueue("control");
	auto color_queue = device.getOutputQueue("color", 4, false);
	auto nn_queue = device.getOutputQueue("nn_out", 4, false);


	std::vector<dai::ImgDetection> detections;

	// Add bounding boxes and text to the frame and show it to the user
	auto displayFrame = [](std::string name, cv::Mat frame, std::vector<dai::SpatialImgDetection>& detections) {
		auto color = cv::Scalar(255, 0, 0);
		auto width = frame.cols;
		auto height = frame.rows;
		auto closest_dist = 99999999;
		// nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
		for (auto& detection : detections) {
			auto coords = detection.spatialCoordinates;
			int x1 = detection.xmin * width;
			int y1 = detection.ymin * height;
			int x2 = detection.xmax * width;
			int y2 = detection.ymax * height;

			auto distance = std::sqrt(coords.x * coords.x + coords.y * coords.y + coords.z * coords.z);
			if (distance < closest_dist)
				closest_dist = distance;

			std::string labelStr = "Face Distance: ";
			//if (closest_dist != 99999999) {
				labelStr += std::to_string(distance / 1000);
				auto new_lens_pos = min(lensMax, closest_dist);
				new_lens_pos = max(lensMin, new_lens_pos);
				if (new_lens_pos != lensPos && new_lens_pos != lensMax) {
					lensPos = new_lens_pos;
					/*auto ctrl = dai::CameraControl();
					ctrl.setManualFocus(lensPos);
					controlQ->se*/
				}
			//}
			cv::putText(frame, labelStr, cv::Point(x1 + 10, y1 + 20), cv::FONT_HERSHEY_TRIPLEX, 0.5, color);
			/*std::string lensStr = "Lens position: " + std::to_string(lensPos);
			cv::putText(frame, lensStr, cv::Point(x1 + 10, y1 + 40), cv::FONT_HERSHEY_TRIPLEX, 0.5, color);
			std::stringstream confStr;
			confStr << std::fixed << std::setprecision(2) << detection.confidence * 100;
			cv::putText(frame, confStr.str(), cv::Point(x1 + 10, y1 + 60), cv::FONT_HERSHEY_TRIPLEX, 0.5, color);*/
			cv::rectangle(frame, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), color, cv::FONT_HERSHEY_SIMPLEX);
		}
		// Show the frame
		cv::imshow(name, frame);
	};

	std::shared_ptr<dai::DataOutputQueue> queues[] = {
		device.getOutputQueue("color", 4, false),
		device.getOutputQueue("nn_out", 4, false),
		device.getOutputQueue("depth", 4, false)
	};

	while (true) {
		auto frame = queues[0]->get<dai::ImgFrame>();
		auto nn_in = queues[1]->get<dai::SpatialImgDetections>();
		auto depth = queues[2]->get<dai::ImgFrame>();

		/*auto depthFrame = depth->getFrame();
		cv::pyrDown(depthFrame, depthFrame);
		cv::normalize(depthFrame, depthFrame, NULL, 255, 0, cv::NORM_INF, CV_8UC1);
		cv::equalizeHist(depthFrame, depthFrame);
		cv::applyColorMap(depthFrame, depthFrame, cv::COLORMAP_HOT);*/

		auto height = frame->getHeight();
		auto width = frame->getWidth();

		if (nn_in != nullptr) {
			auto detections = nn_in->detections;
			displayFrame("color", frame->getCvFrame(), detections);
		}

		//// Wait and check if 'q' pressed
		if (cv::waitKey(1) == 'q') return 0;
	}
	return 0;
}


dai::Pipeline CreatePipeline() {
	dai::Pipeline pipeline;

	// Create pipeline
	auto cam = pipeline.create<dai::node::ColorCamera>();
	cam->setPreviewSize(300, 300);
	cam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
	cam->setVideoSize(1080, 1080);
	cam->setInterleaved(false);

	auto controlIn = pipeline.create<dai::node::XLinkIn>();
	controlIn->setStreamName("control");
	controlIn->out.link(cam->inputControl);

	// Get output queue
	auto left = pipeline.create<dai::node::MonoCamera>();
	left->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
	left->setBoardSocket(dai::CameraBoardSocket::LEFT);

	auto right = pipeline.create<dai::node::MonoCamera>();
	right->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
	right->setBoardSocket(dai::CameraBoardSocket::RIGHT);

	auto stereo = pipeline.create<dai::node::StereoDepth>();
	stereo->initialConfig.setConfidenceThreshold(240);
	stereo->setDepthAlign(dai::CameraBoardSocket::RGB);
	stereo->setExtendedDisparity(true);
	left->out.link(stereo->left);
	right->out.link(stereo->right);

	auto cam_xout = pipeline.create<dai::node::XLinkOut>();
	cam_xout->setStreamName("color");
	cam->video.link(cam_xout->input);

	// Try connecting to device and start the pipeline
	//cout("Creating Face Detection Neural Network...");
	std::string nnPath("blobs/face-detection-retail-0004_openvino_2021.4_6shave.blob");

	auto face_det_nn = pipeline.create<dai::node::MobileNetSpatialDetectionNetwork>();
	face_det_nn->setConfidenceThreshold(0.4);
	face_det_nn->setBlobPath(nnPath);
	face_det_nn->setBoundingBoxScaleFactor(0.5);
	face_det_nn->setDepthLowerThreshold(200);
	face_det_nn->setDepthUpperThreshold(3000);

	cam->preview.link(face_det_nn->input);
	stereo->depth.link(face_det_nn->inputDepth);


	auto nn_xout = pipeline.create<dai::node::XLinkOut>();
	nn_xout->setStreamName("nn_out");
	face_det_nn->out.link(nn_xout->input);

	auto pass_xout = pipeline.create<dai::node::XLinkOut>();
	pass_xout->setStreamName("depth");
	face_det_nn->passthroughDepth.link(pass_xout->input);

	return pipeline;
}
