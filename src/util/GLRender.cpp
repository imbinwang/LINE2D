#include "..\..\include\util\GLRender.h"
#include <fstream>

/* projection(intrinsic) related */
CameraCalibration GLRender::m_calibration; // calibrated camera intrinsic parameters
float GLRender::m_nearPlane; // near clipping plane
float GLRender::m_farPlane; // far clipping plane
int GLRender::m_screenWidth; // screen width
int GLRender::m_screenHeight; // screen height	
GLfloat GLRender::m_projectionMatrix[16]; // Buffered projection matrix.

/* model view (extrinsic) related */
float GLRender::m_radius, GLRender::m_latitude, GLRender::m_longitude, GLRender::m_inplane;
float GLRender::m_minRadius, GLRender::m_maxRadius, GLRender::m_radiusStep; // ball radius
float GLRender::m_minLatitude, GLRender::m_maxLatitude, GLRender::m_latitudeStep; // camera lagitude (in degree)
float GLRender::m_minLongitude, GLRender::m_maxLongitude, GLRender::m_longitudeStep; // camera longitude (in degree)
float GLRender::m_minInplane, GLRender::m_maxInplane, GLRender::m_inplaneStep; // camera inplane (in degree)
float GLRender::m_rotation[3]; // coordinate system axises represented in Rodrigus format
float GLRender::m_translation[3]; // coordinate system origin position
GLfloat GLRender::m_modelViewMatrix[16]; // Buffered model view matrix.

/* the object to be rendered */
GLMmodel *GLRender::m_model;

/* rendered buffer data related */
uchar* GLRender::m_pixelBuffer; // buffered rendered pixels
float* GLRender::m_depthBuffer; // buffered rendered depth data

/* storage for each frame and camera pose during camera viewpoint sampling */
//std::vector<cv::Mat> GLRender::m_viewImgs;
//std::vector<cv::Matx61f> GLRender::m_viewPoses;
std::string GLRender::m_outDir;

static unsigned int viewCounter;
static bool startStoreViewImageAndPose;
static std::string objName;
static std::ofstream poseOutFile;

void GLRender::init(int argc, char **argv,
	GLMmodel *model,
	CameraCalibration& calibration, 
	float nearPlane, float farPlane,
	float minRadius, float maxRadius, float radiusStep,
	float minLatitude, float maxLatitude, float latitudeStep,
	float minLongitude, float maxLongitude, float longitudeStep,
	float minInplane, float maxInplane, float inplaneStep,
	const std::string &out_dir)
{
	m_calibration = calibration;
	m_screenWidth = int(2*m_calibration.cx()+0.5f);
	m_screenHeight = int(2*m_calibration.cy()+0.5f);
	m_nearPlane = nearPlane;
	m_farPlane = farPlane;
	m_pixelBuffer = new uchar[m_screenWidth*m_screenHeight*3];
	m_depthBuffer = new float[m_screenWidth*m_screenHeight*3];
	m_model = model;

	m_minRadius = minRadius; m_maxRadius = maxRadius; m_radiusStep = radiusStep;
	m_minLatitude = minLatitude; m_maxLatitude = maxLatitude; m_latitudeStep = latitudeStep;
	m_minLongitude = minLongitude; m_maxLongitude = maxLongitude; m_longitudeStep = longitudeStep;
	m_minInplane = minInplane; m_maxInplane = maxInplane; m_inplaneStep = inplaneStep;

	unsigned int viewPointsNum = (static_cast<unsigned int>((m_maxRadius - m_minRadius)/ m_radiusStep) + 1)* 
		(static_cast<unsigned int>((m_maxLatitude - m_minLatitude)/ m_latitudeStep) + 1)*
		(static_cast<unsigned int>((m_maxLongitude - m_minLongitude)/ m_longitudeStep) + 1)*
		(static_cast<unsigned int>((m_maxInplane - m_minInplane)/ m_inplaneStep) + 1);
	/*m_viewImgs.resize(viewPointsNum);
	m_viewPoses.resize(viewPointsNum);*/
	m_outDir = out_dir;

	std::string pathname = std::string(m_model->pathname);
	size_t dot_idx = pathname.find_last_of('.');
	size_t slash_idx = pathname.find_last_of("/\\");
	objName = pathname.substr(slash_idx+1, dot_idx);
	std::string out_pose_file_path = m_outDir + "/" + objName + "PoseInRodrigus.txt";
	poseOutFile.open(out_pose_file_path.c_str());

	viewCounter = 0;
	startStoreViewImageAndPose = 0;

	m_radius = m_minRadius;
	m_latitude = m_minLatitude;
	m_longitude = m_minLongitude;
	m_inplane = m_minInplane;

	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH);
	glutInitWindowSize(m_screenWidth, m_screenHeight);
	glutCreateWindow("GLRenderWin");
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutTimerFunc(10, moveCamera, 1);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	glutMainLoop();
}

void GLRender::computeProjectionMatrix()
{
	// Camera parameters
	float f_x = m_calibration.fx(); // Focal length in x axis
	float f_y = m_calibration.fy(); // Focal length in y axis (usually the same)
	float c_x = m_calibration.cx(); // Camera primary point x
	float c_y = m_calibration.cy(); // Camera primary point y


	m_projectionMatrix[0] = 2.0f * f_x / m_screenWidth;
	m_projectionMatrix[1] = 0.0f;
	m_projectionMatrix[2] = 0.0f;
	m_projectionMatrix[3] = 0.0f;


	m_projectionMatrix[4] = 0.0f;
	m_projectionMatrix[5] = 2.0f * f_y / m_screenHeight;
	m_projectionMatrix[6] = 0.0f;
	m_projectionMatrix[7] = 0.0f;


	m_projectionMatrix[8] = 2.0f * c_x / m_screenWidth - 1.0f;
	m_projectionMatrix[9] = 2.0f * c_y / m_screenHeight - 1.0f;    
	m_projectionMatrix[10] = -( m_farPlane + m_nearPlane) / ( m_farPlane - m_nearPlane );
	m_projectionMatrix[11] = -1.0f;


	m_projectionMatrix[12] = 0.0f;
	m_projectionMatrix[13] = 0.0f;
	m_projectionMatrix[14] = -2.0f * m_farPlane * m_nearPlane / ( m_farPlane - m_nearPlane );        
	m_projectionMatrix[15] = 0.0f;
}

void GLRender::computeModelViewMatrix()
{
	cv::Matx31f rvec(m_rotation);
	cv::Matx33f rmat;
	cv::Rodrigues(rvec, rmat);
	rmat = rmat.inv();

	m_modelViewMatrix[0] = rmat.val[0];
	m_modelViewMatrix[1] = rmat.val[3];
	m_modelViewMatrix[2] = rmat.val[6];
	m_modelViewMatrix[3] = 0.0f;

	m_modelViewMatrix[4] = rmat.val[1];
	m_modelViewMatrix[5] = rmat.val[4];
	m_modelViewMatrix[6] = rmat.val[7];
	m_modelViewMatrix[7] = 0.0f;

	m_modelViewMatrix[8] = rmat.val[2];
	m_modelViewMatrix[9] = rmat.val[5];
	m_modelViewMatrix[10] = rmat.val[8];
	m_modelViewMatrix[11] = 0.0f;

	cv::Matx31f tvec(m_translation);
	tvec = rmat*tvec;

	m_modelViewMatrix[12] = -tvec.val[0];
	m_modelViewMatrix[13] = -tvec.val[1];
	m_modelViewMatrix[14] = -tvec.val[2];
	m_modelViewMatrix[15] = 1.0f;
}

void GLRender::loadProjectionMatrix(bool reset)
{
	glMatrixMode(GL_PROJECTION);

	if (reset)
		glLoadIdentity();

	computeProjectionMatrix();
	glMultMatrixf(m_projectionMatrix);
}

void GLRender::loadModelViewMatrix(bool reset)
{
	glMatrixMode(GL_MODELVIEW);
	computeModelViewMatrix();
	if (reset)
		glLoadMatrixf(m_modelViewMatrix);
	else
		glMultMatrixf(m_modelViewMatrix);
}


void GLRender::display()
{
	if(m_model)
	{
		glEnable(GL_LIGHT0);
		glEnable(GL_LIGHTING);
		glEnable(GL_DEPTH_TEST);

		glClearColor(0.0f,0.0f,0.0f,1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

		glPushMatrix();
		loadModelViewMatrix();
		glmDraw(m_model, GLM_SMOOTH|GLM_MATERIAL);
		glPopMatrix();

		glFlush();

		if(startStoreViewImageAndPose)
		{
			// render the rendered color buffer
			// and change the camera pose to object pose
			cv::Mat viewImage = getRenderedColorImage();
			cv::Matx33f cameraCoorSystem = cv::Matx33f::eye();
			cv::Matx31f cameraCoorSystem_rodrigues;
			cameraCoorSystem_rodrigues(0) = m_rotation[0];
			cameraCoorSystem_rodrigues(1) = m_rotation[1];
			cameraCoorSystem_rodrigues(2) = m_rotation[2]; 
			cv::Rodrigues(cameraCoorSystem_rodrigues, cameraCoorSystem);
			cv::Matx31f cameraCoorOrigin = cv::Matx31f::zeros();
			cameraCoorOrigin(0) = m_translation[0];
			cameraCoorOrigin(1) = m_translation[1];
			cameraCoorOrigin(2) = m_translation[2];

			cv::Matx33f objectCoorSystem = cv::Matx33f::eye();
			cv::Matx31f objectCoorSystem_rodrigues;
			objectCoorSystem = cameraCoorSystem.inv();
			cv::Rodrigues(objectCoorSystem, objectCoorSystem_rodrigues);
			cv::Matx31f objectCoorOrigin = cv::Matx31f::zeros();
			objectCoorOrigin = -objectCoorSystem*cameraCoorOrigin;

			cv::Matx61f objectPose;
			objectPose(0) = objectCoorSystem_rodrigues(0);
			objectPose(1) = objectCoorSystem_rodrigues(1);
			objectPose(2) = objectCoorSystem_rodrigues(2);
			objectPose(3) = objectCoorOrigin(0);
			objectPose(4) = objectCoorOrigin(1);
			objectPose(5) = objectCoorOrigin(2);

			// save the view image and corresponding pose
			/*m_viewImgs[viewCounter-1] = viewImage;
			m_viewPoses[viewCounter-1] = objectPose;*/
			char buf_[10];
			sprintf(buf_, "%d", viewCounter-1);
			std::string vc_ = buf_;
			std::string out_img_file_path = m_outDir + "/" + objName + vc_ + ".png";
			cv::imwrite(out_img_file_path, viewImage);
			poseOutFile << objectPose.val[0] << objectPose.val[1] << objectPose.val[2]
			<< objectPose.val[3] << objectPose.val[4] << objectPose.val[5] << "\n";

			startStoreViewImageAndPose = 0;
		}
	}
} 

struct GLRenderVec
{
	float x,y,z;
};
inline float vecDot(const GLRenderVec &vec1, const GLRenderVec &vec2)
{
	return vec1.x*vec2.x+vec1.y*vec2.y+vec1.z*vec2.z;
}
inline GLRenderVec vecCross(const GLRenderVec &vec1, const GLRenderVec &vec2)
{
	GLRenderVec ret;
	ret.x = vec1.y*vec2.z - vec1.z*vec2.y;
	ret.y = -vec1.x*vec2.z + vec1.z*vec2.x;
	ret.z = vec1.x*vec2.y - vec1.y*vec2.x;

	return ret;
}
inline float vecNorm(const GLRenderVec &vec1)
{
	return sqrt(vec1.x*vec1.x + vec1.y*vec1.y + vec1.z*vec1.z);
}
void GLRender::moveCamera(int value)
{
	startStoreViewImageAndPose = 1;

	// define camera coordinate system
	GLRenderVec x_axis, y_axis, z_axis, up;
	x_axis.x = 1; x_axis.y = 0; x_axis.z = 0;
	y_axis.x = 0; y_axis.y = 1; y_axis.z = 0;
	z_axis.x = 0; z_axis.y = 0; z_axis.z = 1;
	up.x = 0; up.y = 0; up.z = 1;
	cv::Matx33f cs = cv::Matx33f::eye();

	// change scale
	if(m_radius<=m_maxRadius)
	{
		// change latitude
		if(m_latitude<=m_maxLatitude)
		{
			float lati_radian = m_latitude*PI/180;

			// when at north polar or south polar, ignore the longitude change
			if(m_latitude==90.0f || m_latitude==-90.0f)
			{
				// camera origin location
				m_translation[0] = 0.0f;
				m_translation[1] = 0.0f;
				m_translation[2] = m_radius;

				// change inplane rotation
				if(m_inplane<=m_maxInplane)
				{
					// view direction axis
					float axis[3] = {-m_translation[0]/m_radius, -m_translation[1]/m_radius, -m_translation[2]/m_radius};

					// calculate camera coordinate system axises
					z_axis.x = -axis[0]; z_axis.y = -axis[1]; z_axis.z = -axis[2];
					x_axis.x = 1; x_axis.y = 0; x_axis.z = 0;
					y_axis.x = 0; y_axis.y = 1; y_axis.z = 0;
					cs(0,0) = x_axis.x; cs(0,1) = y_axis.x; cs(0,2) = z_axis.x;
					cs(1,0) = x_axis.y; cs(1,1) = y_axis.y; cs(1,2) = z_axis.y; 
					cs(2,0) = x_axis.z; cs(2,1) = y_axis.z; cs(2,2) = z_axis.z; 

					// define inplate rotation matrix
					// rotation alone the z-axis (i.e. the inversed view direction)
					float inplane_radian = m_inplane*PI/180;
					cv::Matx33f inplaneRot = cv::Matx33f::eye();
					inplaneRot(0,0) = cos(inplane_radian); inplaneRot(0,1) = -sin(inplane_radian);
					inplaneRot(1,0) = sin(inplane_radian); inplaneRot(1,1) = cos(inplane_radian);
					cs = cs*inplaneRot;

					// represent the camera coordinate system using Rodrigues formate
					cv::Matx31f cs_rodrigues;
					cv::Rodrigues(cs, cs_rodrigues);
					m_rotation[0] = cs_rodrigues(0);
					m_rotation[1] = cs_rodrigues(1);
					m_rotation[2] = cs_rodrigues(2);

					// go to next inplane rotation state
					m_inplane+=m_inplaneStep;
					++viewCounter;

					// animation
					glutPostRedisplay();
					glutTimerFunc(10, moveCamera, 1);
					return;
				}
			}
			else
			{
				// change longitude
				if(m_longitude<=m_maxLongitude)
				{
					float longi_radian = m_longitude*PI/180;

					// define camera origin location
					m_translation[0] = m_radius * cos(lati_radian) * cos(longi_radian);
					m_translation[1] = m_radius * cos(lati_radian) * sin(longi_radian);
					m_translation[2] = m_radius * sin(lati_radian);

					// change inplane rotation
					if(m_inplane<=m_maxInplane)
					{
						// view direction axis
						float axis[3] = {-m_translation[0]/m_radius, -m_translation[1]/m_radius, -m_translation[2]/m_radius};				

						// calculate camera coordinate system axises
						z_axis.x = -axis[0]; z_axis.y = -axis[1]; z_axis.z = -axis[2];
						x_axis = vecCross(up, z_axis);
						float x_axis_norm = vecNorm(x_axis);
						if(x_axis_norm>0)
							x_axis.x /= x_axis_norm; x_axis.y /= x_axis_norm; x_axis.z /= x_axis_norm;
						y_axis = vecCross(z_axis, x_axis);
						float y_axis_norm = vecNorm(y_axis);
						if(y_axis_norm>0)
							y_axis.x /= y_axis_norm; y_axis.y /= y_axis_norm; y_axis.z /= y_axis_norm;

						cs(0,0) = x_axis.x; cs(0,1) = y_axis.x; cs(0,2) = z_axis.x;
						cs(1,0) = x_axis.y; cs(1,1) = y_axis.y; cs(1,2) = z_axis.y; 
						cs(2,0) = x_axis.z; cs(2,1) = y_axis.z; cs(2,2) = z_axis.z; 

						// define inplate rotation matrix
						// rotation alone the z-axis (i.e. the inversed view direction)
						float inplane_radian = m_inplane*PI/180;
						cv::Matx33f inplaneRot = cv::Matx33f::eye();
						inplaneRot(0,0) = cos(inplane_radian); inplaneRot(0,1) = -sin(inplane_radian);
						inplaneRot(1,0) = sin(inplane_radian); inplaneRot(1,1) = cos(inplane_radian);
						cs = cs*inplaneRot;

						// represent the camera coordinate system using Rodrigues formate
						cv::Matx31f cs_rodrigues;
						cv::Rodrigues(cs, cs_rodrigues);
						m_rotation[0] = cs_rodrigues(0);
						m_rotation[1] = cs_rodrigues(1);
						m_rotation[2] = cs_rodrigues(2);

						// go to next inplane rotation state
						m_inplane+=m_inplaneStep;
						++viewCounter;

						// animation
						glutPostRedisplay();
						glutTimerFunc(10, moveCamera, 1);
						return;
					}

					m_longitude+=m_longitudeStep;
					m_inplane = m_minInplane;

					// animation
					glutTimerFunc(10, moveCamera, 1);
					return;
				}
			}

			m_latitude+=m_latitudeStep;
			m_longitude = m_minLongitude;
			m_inplane = m_minInplane;

			// animation
			glutTimerFunc(10, moveCamera, 1);
			return;
		}

		m_radius+=m_radiusStep;
		m_latitude = m_minLatitude;
		m_longitude = m_minLongitude;
		m_inplane = m_minInplane;

		// animation
		glutTimerFunc(10, moveCamera, 1);
		return;
	}

	// done the camera viewpoints sampling and exit the main loop
	// close the opengl context
	poseOutFile.close();
	glutLeaveMainLoop();
}

void GLRender::reshape(int width, int height)
{
	loadProjectionMatrix();
}

cv::Mat GLRender::getRenderedColorImage()
{
	glReadPixels(0,0,m_screenWidth,m_screenHeight,GL_RGB,GL_UNSIGNED_BYTE,m_pixelBuffer);
	cv::Mat bufferImg(m_screenHeight,m_screenWidth,CV_8UC3,cv::Scalar(0,0,0));
	//copy the data form m_pixelBuffer to cv::Mat
	for(int i=0; i<m_screenHeight; ++i)
	{
		for(int j=0; j<m_screenWidth; ++j)
		{	
			//RGB->BGR
			bufferImg.at<cv::Vec3b>(m_screenHeight-i-1,j)[2] = m_pixelBuffer[i*m_screenWidth*3+3*j];  
			bufferImg.at<cv::Vec3b>(m_screenHeight-i-1,j)[1] = m_pixelBuffer[i*m_screenWidth*3+3*j+1];  
			bufferImg.at<cv::Vec3b>(m_screenHeight-i-1,j)[0] = m_pixelBuffer[i*m_screenWidth*3+3*j+2];  
		}
	}

	return bufferImg;
}
